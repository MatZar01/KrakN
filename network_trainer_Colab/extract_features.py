#!python3
# mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')
# add local libraries to system path

import sys
sys.path.append('./utilities/io')

from hdf5_dataset_writer import HDF5DatasetWriter
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import progressbar
import random
import os

# set paths
datasetPath = r'/content/gdrive/My Drive/KrakN/database'
outputPath = r"/content/gdrive/My Drive/KrakN/database/features"

# set/check dataset & output path, delete previous output if exists
if not os.path.exists(datasetPath):
    print("Dataset at {}\nDoes not exist!\nQuitting now".format(datasetPath))
    quit()

batchSize = 256
bufferSize = 1000

# load images and shuffle them
print("Loading images...")
imagePaths = list(paths.list_images(datasetPath))
random.shuffle(imagePaths)
print("{} images loaded".format(len(imagePaths)))

# get scale factor
splitted = imagePaths[0].split('_')
scale = splitted[-1]
scale = scale[:-4]

# extract class labels and encode them
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load VGG16 network excluding final FC layers
print("loading network...")
model = VGG16(weights="imagenet", include_top=False)

# initialize dataset writer
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), outputPath + "_s_{}".format(scale) + '.hdf5', "features", bufferSize)
dataset.storeClassLabels(le.classes_)

# initialize progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over images
for i in np.arange(0, len(imagePaths), batchSize):
    batchPaths = imagePaths[i:i + batchSize]
    batchLabels = labels[i:i + batchSize]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        # load and resize image
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess image by expanding and subtracting mean RGB value
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add image to batch
        batchImages.append(image)

    # pass images thr network
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batchSize)

    # reshape features
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add features and labels to dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
