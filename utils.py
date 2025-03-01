import random
from typing import List

import tensorflow as tf
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam

from config import CATEGORIES, JSON_DATA
import cv2
import json
import uuid
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import warnings
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from keras import initializers, regularizers


class GenericObject:
    """
    Generic object data.
    """
    def __init__(self):
        self.id = uuid.uuid4()
        self.bb = (-1, -1, -1, -1)
        self.category= -1
        self.score = -1

class GenericImage:
    """
    Generic image data.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tile = np.array([-1, -1, -1, -1])  # (pt_x, pt_y, pt_x+width, pt_y+height)
        self.objects = list([])

    def add_object(self, obj: GenericObject):
        self.objects.append(obj)

def load_geoimage(filename):
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    src_raster = rasterio.open('xview_recognition/'+filename, 'r')
    # RasterIO to OpenCV (see inconsistencies between libjpeg and libjpeg-turbo)
    input_type = src_raster.profile['dtype']
    input_channels = src_raster.count
    img = np.zeros((src_raster.height, src_raster.width, src_raster.count), dtype=input_type)
    for band in range(input_channels):
        img[:, :, band] = src_raster.read(band+1)
    return img

def generator_images(objs, batch_size, do_shuffle=False):
    while True:
        if do_shuffle:
            np.random.shuffle(objs)
        groups = [objs[i:i+batch_size] for i in range(0, len(objs), batch_size)]
        for group in groups:
            images, labels = [], []
            for (filename, obj) in group:
                # Load image
                images.append(load_geoimage(filename))
                probabilities = np.zeros(len(CATEGORIES))
                probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
                labels.append(probabilities)
            images = np.array(images).astype(np.float32)
            labels = np.array(labels).astype(np.float32)
            yield images, labels


def augmented_generation(objs, batch_size, do_shuffle=True):
    if do_shuffle:
        np.random.shuffle(objs)

    while True:
        images, labels = [], []
        for obj in objs:
            filename, obj = obj
            image = load_geoimage(filename)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            # Rotate by a small random angle (e.g., between -2 and 2 degrees)
            angle = random.uniform(-2, 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))

            # Crop a small margin from the edges (adjust the margin as needed)
            margin = random.uniform(0, 10)  # number of pixels to crop from each side
            image = rotated[margin:h - margin, margin:w - margin]

            # Resize the cropped image back to original size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

            delta = np.round(np.random.uniform(size=(3,), low=5, high=5))
            image[0, :] += delta[0]
            image[1, :] += delta[1]
            image[2, :] += delta[2]
            images.append(image)


            probabilities = np.zeros(len(CATEGORIES))
            probabilities[list(CATEGORIES.values()).index(obj.category)] = 1
            labels.append(probabilities)
            if len(images) == batch_size:
                images = np.array(images).astype(np.float32)
                labels = np.array(labels).astype(np.float32)
                yield images, labels



def augment_image(image, crop_ratio=0.8):
    """
    Augments a single image by applying rotations and center crops.

    Parameters:
        image (np.array): The input image array.
        crop_ratio (float): The fraction of the image to keep when cropping (default 0.8).

    Returns:
        List[np.array]: A list of augmented image arrays.
    """
    augmented = []
    # Apply rotations: 0, 90, 180, 270 degrees.
    for k in range(4):
        # Rotate the image by 90 degrees k times.
        rotated = np.rot90(image, k=k)
        augmented.append(rotated)

        # Compute dimensions for center crop.
        h, w = rotated.shape[:2]
        crop_h, crop_w = int(crop_ratio * h), int(crop_ratio * w)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = rotated[start_y:start_y + crop_h, start_x:start_x + crop_w]
        augmented.append(cropped)
    return augmented


def augment_batch(images, labels, crop_ratio=0.8):
    """
    Augments a batch of images and duplicates the labels accordingly.

    Parameters:
        images (List[np.array]): List or array of images.
        labels (List[np.array]): List or array of corresponding labels.
        crop_ratio (float): The fraction of the image to keep when cropping.

    Returns:
        Tuple[np.array, np.array]: Augmented images and labels.
    """
    aug_images = []
    aug_labels = []
    for img, lab in zip(images, labels):
        augmented_imgs = augment_image(img, crop_ratio)
        aug_images.extend(augmented_imgs)
        aug_labels.extend([lab] * len(augmented_imgs))
    return np.array(aug_images, dtype=np.float32), np.array(aug_labels, dtype=np.float32)



def draw_confusion_matrix(cm, categories):
    # Draw confusion matrix
    fig = plt.figure(figsize=[6.4*pow(len(categories), 0.5), 4.8*pow(len(categories), 0.5)])
    ax = fig.add_subplot(111)
    cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=list(categories.values()), yticklabels=list(categories.values()), ylabel='Annotation', xlabel='Prediction')
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=int(20-pow(len(categories), 0.5)))
    fig.tight_layout()
    plt.show(fig)


def generate_predictions(model, out_name="predictions"):
    counts = dict.fromkeys(CATEGORIES.values(), 0)
    anns = []
    for json_img, json_ann in zip(JSON_DATA['images'].values(), JSON_DATA['annotations'].values()):
        image = GenericImage(json_img['filename'])
        image.tile = np.array([0, 0, json_img['width'], json_img['height']])
        obj = GenericObject()
        obj.bb = (
        int(json_ann['bbox'][0]), int(json_ann['bbox'][1]), int(json_ann['bbox'][2]), int(json_ann['bbox'][3]))
        obj.category = json_ann['category_id']
        # Resampling strategy to reduce training time
        counts[obj.category] += 1
        image.add_object(obj)
        anns.append(image)
    print(counts)


    model.load_weights('model.keras', by_name=True)
    predictions_data = {"images": {}, "annotations": {}}
    for idx, ann in enumerate(anns):
        image_data = {"image_id": ann.filename.split('/')[-1], "filename": ann.filename, "width": int(ann.tile[2]),
                      "height": int(ann.tile[3])}
        predictions_data["images"][idx] = image_data
        # Load image
        image = load_geoimage(ann.filename)
        for obj_pred in ann.objects:
            # Generate prediction
            warped_image = np.expand_dims(image, 0)
            predictions = model.predict(warped_image, verbose=0)
            # Save prediction
            pred_category = list(CATEGORIES.values())[np.argmax(predictions)]
            pred_score = np.max(predictions)
            annotation_data = {"image_id": ann.filename.split('/')[-1], "category_id": pred_category,
                               "bbox": [int(x) for x in obj_pred.bb]}
            predictions_data["annotations"][idx] = annotation_data

    with open(f"{out_name}.json", "w") as outfile:
        json.dump(predictions_data, outfile)

    zip_filename = f"{out_name}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(f"{out_name}.json")


class LayerSettings:
    def __init__(self, neurons=512, penalization=None, activation='elu', initialization='He', dropout=0, batch_normalization=True):
        self.neurons = neurons
        self.penalization = penalization
        self.activation = activation
        self.initialization = initialization
        self.batch_normalization = batch_normalization
        self.dropout = dropout


def generate_model(layers : List[LayerSettings]=(),
                   input=Flatten(input_shape=(244, 244, 3)),
                   optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True, clipnorm=1.0)):
    model = Sequential()
    model.add(input)

    for settings in layers:
        if settings.initialization.lower() == 'he':
            init = initializers.HeNormal()
        elif settings.initialization.lower() == 'glorot':
            init = initializers.GlorotUniform()
        else:
            init = initializers.RandomNormal()
        # Set regularizer if penalization is provided.
        reg = regularizers.l2(settings.penalization) if settings.penalization is not None else None
        activation = Activation(settings.activation)

        model.add(Dense(settings.neurons, kernel_initialization=init, kernel_regularization=reg))
        if settings.batch_normalization:
            model.add(BatchNormalization())
        model.add(activation)
        if settings.dropout > 0:
            model.add(Dropout(settings.dropout))

    model.add(Dense(len(CATEGORIES), activation='softmax'))

    # Learning rate is changed to 0.001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', ' precision'])
    return model