#!python3
try:
    import platform
    import cv2
    import os
    from keras.applications import VGG16
    from keras.preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    import pickle
    import h5py
    import numpy as np
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

# set input image, mask & model path
IMAGE_DIR = r'.{}Images'.format(os.path.sep, os.path.sep)
MASK_DIR = r'.{}Masks'.format(os.path.sep, os.path.sep)
MODEL_DIR = r'.{}Classifiers'.format(os.path.sep)
if not os.path.exists(IMAGE_DIR):
    print("No input image provided!")
    os.makedirs(IMAGE_DIR)
    quit()
if not os.path.exists(MASK_DIR):
    print("No input mask provided!")
    os.makedirs(MASK_DIR)
    quit()
if not os.path.exists(MODEL_DIR):
    print("No models provided!")
    os.makedirs(MODEL_DIR)
    quit()
imagePaths = list(paths.list_images(IMAGE_DIR))
if len(imagePaths) == 0:
    print("No input image provided!")
    quit()

# set the rest of paths
outputPathDir = r".{}Output".format(os.path.sep)
imageDir = outputPathDir + os.path.sep + "Images"
maskDir = outputPathDir + os.path.sep + "Masks"

# check files integrity
if not os.path.exists(outputPathDir):
    os.mkdir(outputPathDir)
if not os.path.exists(imageDir):
    os.mkdir(imageDir)
if not os.path.exists(maskDir):
    os.mkdir(maskDir)

# set scale 
scale = 2

# load feature extractor
f_e = VGG16(weights="imagenet", include_top=False)

classifier_list = list(paths.list_files(MODEL_DIR))
image_list = list(paths.list_images(IMAGE_DIR))
mask_list = list(paths.list_images(MASK_DIR))

# set tile window & step sizes
tile_size = 224
windowSize = int(tile_size // scale)
pxStep = 1

for image_path in image_list:
    #load image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # get image format length
    image_name = imagePath.split('.')[-1]
    image_format_len = len(image_format + 1)

    # find corresponding masks
    masks_paths = []
    image_base = image_path.split(os.path.sep)[-1]
    image_base = image_base.split('_')[0]
    for mask_path in mask_list:
        if image_base == mask_path.split(os.path.sep)[-1].split('_')[0]:
            masks_paths.append(mask_path)

    if len(masks_paths) == 0:
        print('No masks found for {} image. Skipping.'.format(image_path))
        continue

    for mask_path in masks_paths:
        # load mask
        mask = cv2.imread(mask_path)

        # start from top
        xTop = 0
        yTop = 0

        # set output
        out = np.zeros((h, w))

        # find classifier for mask
        model = None
        mask_name = mask_path.split(os.path.sep)[-1].split('_')[1]
        for classifier_path in classifier_list:
            if classifier_path.split(os.path.sep)[-1].split('.')[0] in mask_name:
                model = pickle.load(open(classifier_path, 'rb'))
        if model == None:
            print('No model for {} mask found. Skipping.'.format(mask_path.split(os.path.sep)[-1]))
            continue
        
        # load classes and get number of classes
        classes = model.classes_
        class_number = len(model.classes_)

        # add output files
        outputs = []
        for _ in range(0, class_number):
            outputs.append(np.zeros((h,w)))

        # last column & row indicators:
        lastCol = False
        lastRow = False

        # classify image parts
        while True:
            sub_mask = mask[yTop:yTop + windowSize, xTop:xTop + windowSize]

            # check if proceed with sub image
            if 0 not in sub_mask:
                # predict
                sub_image = image[yTop:yTop + windowSize, xTop:xTop + windowSize]

                # Get image features
                img_data = image.img_to_array(sub_image)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)

                # extract features with VGG16
                features = f_e.predict(img_data)
                features = features.reshape((features.shape[0], 512 * 7 * 7))

                # get prediction
                predict = model.predict_proba(features)
                predictionList = list(predict)[0]

                # assess predictions and add them to output images
                maxPredIndex = np.argmax(predictionList)
                maxPred = predictionList[maxPredIndex]
                cv2.rectangle(outputs[maxPredIndex], (xTop, yTop), (xTop + windowSize, yTop + windowSize), (255, 255, 255), -1)

            # advance image
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
                    # write output masks
                    for i in range(0, class_number):
                        cv2.imwrite(maskDir + os.path.sep + image_path.split(os.path.sep)[-1][:-1*image_format_len] + '_' + str(classes[i]) + "_mask" + ".png", outputs[i])
                    print("All parts complete!")
                    break
        
        print("Exporting original image...")
        cv2.imwrite(imageDir + os.path.sep + image_path.split(os.path.sep)[-1][:-1*image_format_len] + '_' + str(classes[i]) + "_orig" + ".png", image)
        print("Saving {} masked images".format(class_number))

        # add bounding boxes to input image
        for i in range(0, class_number):
            # turn mask to grayscale, get contours, copy input image
            contours, _ = cv2.findContours(outputs[i].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            outputImage = image.copy()

            # draw each contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(outputImage, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # write output image
            cv2.imwrite(outputPathDir + os.path.sep + image_path.split(os.path.sep)[-1][:-1*image_format_len] + '_' + str(classes[i]) + "_out" + ".png", outputImage)
            print("{} of {} images done!".format(i, class_number))
