#!python3
# mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

import cv2
from os import listdir
from os.path import isfile, join
import os
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pickle
import h5py
import numpy as np
import cv2
from imutils import paths

# set paths
modelPath = r"/content/gdrive/My Drive/KrakN/KrakN.cpickle"
imagePath = r"/content/gdrive/My Drive/KrakN/input/2.png"
databasePath = r"/content/gdrive/My Drive/KrakN/database/features_s_2.hdf5"
outputPathDir = r"/content/gdrive/My Drive/KrakN/output"

# check files integrity
input_paths = [modelPath, imagePath, databasePath, outputPathDir]
for input_path in input_paths:
    if not os.path.exists(input_path):
        print("Input path {} not found!\nQuitting now".format(input_path))
        quit()

# set confidence threshold
confidenceThreshold = 0.95

# load models
classifier = pickle.load(open(modelPath, 'rb'))
model = VGG16(weights="imagenet", include_top=False)

# load labels from database
db = h5py.File(databasePath, "r")
labelsDb = db['label_names']
labels = []
for label in labelsDb: labels.append(label)

# load scale factor from database
splitted = databasePath.split('_')
scale = splitted[-1]
scale = float(databasePath.split('_')[-1].replace('.' + path.split('.')[-1], ''))

# set tile window & step sizes
overlap = 0.6
tileSize = 224
windowSize = int(tileSize // scale)
pxStep = int(windowSize * (1 - overlap))

# load image
imageCV = cv2.imread(imagePath)
(h, w) = imageCV.shape[:2]

# add mask images
maskList = []
for label in labels:
    maskList.append(np.zeros((h, w)))

# compute number of tiles for image
tilesNumber = int(((w - (windowSize - pxStep)) / pxStep)) * int(((h - (windowSize - pxStep)) / pxStep))
xTop = 0
yTop = 0

# initialize counter
counter = 0

# last column & row indicators:
lastCol = False
lastRow = False

while True:
    # crop single tile from image
    subImage = imageCV[yTop:yTop + windowSize, xTop:xTop + windowSize]

    # resize sub image according to database settings
    subImage = cv2.resize(subImage, (tileSize, tileSize), interpolation=cv2.INTER_CUBIC)

    # Get image features
    img_data = image.img_to_array(subImage)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # extract features with VGG16
    features = model.predict(img_data)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # get prediction
    predict = classifier.predict_proba(features)
    predictionList = list(predict)[0]

    # assess predictions and add them to output images
    maxPredIndex = np.argmax(predictionList)
    maxPred = predictionList[maxPredIndex]
    if maxPred >= confidenceThreshold:
        cv2.rectangle(maskList[maxPredIndex], (xTop, yTop), (xTop + windowSize, yTop + windowSize), (255, 255, 255), -1)

    counter += 1

    if counter % 100 == 0:
        print("{} out of {} parts complete".format(counter, tilesNumber))

    #print("Image classified as {}".format(labels[int(predict)]))

    xTop += pxStep
    if xTop + windowSize >= w:
        if lastCol == False:
            lastCol = True
            xTop = w - windowSize - 1
        else:
            lastCol = False
            xTop = 0
            yTop += pxStep

    if yTop + windowSize >= h:
        if lastRow == False:
            lastRow = True
            yTop = h - windowSize - 1
        else:
            for i in range(0, len(labels)):
                cv2.imwrite(outputPathDir + '/' + imagePath.split('/')[-1].split('.')[0] + '_' + labels[i] + "_mask" + ".png", maskList[i])
            print("All parts complete!")
            break

print("Saving {} masked images".format(len(labels)))

# add bounding boxes to input image
for i in range(0, len(labels)):
    # turn mask to grayscale, get contours, copy input image
    contours, _ = cv2.findContours(maskList[i].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outputImage = imageCV.copy()

    # draw each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(outputImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # write output image
    cv2.imwrite(outputPathDir + '/' + imagePath.split('/')[-1].split('.')[0] + '_' + labels[i] + "_out" + ".png", outputImage)
    print("{} of {} images done!".format(i, len(labels)))
