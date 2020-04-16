#!python3
try:
    import platform
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
except ImportError as e:
    print(e)
    print("One or more dependencies missing!\nOpen README file to check required dependencies.")
    if platform.system() == 'Linux':
        print("\nYou can install all dependencies using 'sudo chmod +x ./install_dependencies.sh & ./install_"
              "dependencies.sh' command in KrakN directory.")
    else:
        print("\nYou can install all dependencies using install_dependencies.bat in KrakN directory")
    quit()

# set input image path
IMAGE_DIR = r'.{}input'.format(os.path.sep)
if not os.path.exists(IMAGE_DIR):
    print("No input image provided!")
    os.mkdir(IMAGE_DIR)
    quit()
imagePaths = list(paths.list_images(IMAGE_DIR))
if len(imagePaths) == 0:
    print("No input image provided!")
    quit()

# set the rest of paths
modelPath = r'.{}KrakN_model.cpickle'.format(os.path.sep)
databasePath = r".{}database".format(os.path.sep)
outputPathDir = r".{}output".format(os.path.sep)
imageDir = outputPathDir + os.path.sep + "Images"
maskDir = outputPathDir + os.path.sep + "Masks"

# check files integrity
if not os.path.exists(outputPathDir):
    os.mkdir(outputPathDir)
if not os.path.exists(imageDir):
    os.mkdir(imageDir)
if not os.path.exists(maskDir):
    os.mkdir(maskDir)

# search for features file
file_found = False
feature_files_number = 0
database_list_files = os.listdir(databasePath)
for file_path in database_list_files:
    if 'features' in file_path:
        feature_files_number += 1
        databasePath += os.path.sep
        databasePath += file_path
        file_found = True
if not file_found:
    print("Features file at {}\nDoes not exist!\nQuitting now".format(databasePath))
    quit()
if feature_files_number != 1:
    print("There can be only 1 features file in database directory while there are {}\nRemove excessive files".format(feature_files_number))
    quit()

input_paths = [modelPath]
for input_path in input_paths:
    if not os.path.exists(input_path):
        print("Path {} not found!\nQuitting now".format(input_path))
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
scale = int(scale[:-5])

# set tile window & step sizes
overlap = 0.6
tileSize = 224
windowSize = int(tileSize // scale)
pxStep = int(windowSize * (1 - overlap))

for imagePath in imagePaths:
    # load image
    imageCV = cv2.imread(imagePath)
    (h, w) = imageCV.shape[:2]

    # get image format length
    image_name = imagePath.split('.')[-1]
    image_format_len = len(image_format + 1)

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
                    cv2.imwrite(maskDir + os.path.sep + imagePath.split(os.path.sep)[-1][:-1*image_format_len] + '_' + labels[i] + "_mask" + ".png", maskList[i])
                print("All parts complete!")
                break

    print("Exporting original image...")
    cv2.imwrite(imageDir + os.path.sep + imagePath.split(os.path.sep)[-1][:-1*image_format_len] + '_' + labels[i] + "_orig" + ".png", imageCV)
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
        cv2.imwrite(outputPathDir + os.path.sep + imagePath.split(os.path.sep)[-1][:-1*image_format_len] + '_' + labels[i] + "_out" + ".png", outputImage)
        print("{} of {} images done!".format(i, len(labels)))
