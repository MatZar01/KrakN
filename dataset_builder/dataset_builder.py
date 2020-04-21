#!python3
try:
    import platform
    import cv2
    from imutils import paths
    import numpy as np
    import pygame
    import os
    import itertools
    import random
	import sys
except ImportError as e:
    print(e)
    print("One or more dependencies missing!\nOpen README file to check required dependencies.")
    if platform.system() == 'Linux':
        print("\nYou can install all dependencies using 'sudo chmod +x ./install_dependencies.sh & ./install_"
              "dependencies.sh' command in KrakN directory.")
    else:
        print("\nYou can install all dependencies using install_dependencies.bat in KrakN directory")
    sys.exit()

# def 'on_trackbar' handlers to manage window position:
def on_trackbar_horizontal(val):
    global pos_x
    pos_x = cv2.getTrackbarPos(TRACKBAR_HORIZONTAL, WINDOW_NAME)
    crop = image[pos_y:pos_y + screen_height, pos_x:pos_x + screen_width]
    cv2.imshow(WINDOW_NAME, crop)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

def on_trackbar_vertical(val):
    global pos_y
    global crop
    pos_y = cv2.getTrackbarPos(TRACKBAR_VERTICAL, WINDOW_NAME)
    crop = image[pos_y:pos_y + screen_height, pos_x:pos_x + screen_width]
    cv2.imshow(WINDOW_NAME, crop)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

# def on_click mouse callback to point defects polylines
def on_mouse(event, x, y, flags, param):
    global global_centers
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append((x + pos_x, y + pos_y))
        if len(point_list) == 2:
            cv2.line(image, point_list[0], point_list[1], (0, 0, 255), 2)
            x_step = (point_list[1][0] - point_list[0][0]) // (FRAME_COUNT + 1)
            y_step = (point_list[1][1] - point_list[0][1]) // (FRAME_COUNT + 1)
            if len(global_centers) == 0:
                global_centers.append(point_list[0])
            for i in range(1, FRAME_COUNT + 1):
                cv2.circle(image, (point_list[0][0] + i * x_step, point_list[0][1] + i * y_step), 3, (0, 0, 255), -1)
                global_centers.append((point_list[0][0] + i * x_step, point_list[0][1] + i * y_step))
            global_centers.append(point_list[1])
            point_list.pop(0)
        cv2.circle(image, (x+pos_x, y+pos_y), 5, (0, 0, 255), -1)

# get screen resolution to fit image crop
pygame.init()
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h - 200
pygame.quit()

# set global constraints
WINDOW_NAME = "Dataset Builder"
TRACKBAR_HORIZONTAL = "Navigate width"
TRACKBAR_VERTICAL = "Navigate height"
FRAME_COUNT = 3
# set zoom factor
while True:
	try:
		ZOOM_FACTOR = float(input("Set zoom factor: "))
		break
	except:
		print("Error, use '.' as decimal separator")
FRAME_SIZE = int(224 // ZOOM_FACTOR)
HALF_SIZE = int(FRAME_SIZE // 2)
os_separator = os.path.sep
DATAPOINTS_STRING = r'.{}datapoints{}'.format(os_separator, os_separator)
IMAGE_FILE_STRING = r'.{}database{}Images'.format(os_separator, os_separator)
FILE_DONE_STRING = IMAGE_FILE_STRING.replace("Images", "DONE")
FILE_ITERATOR = 0
FRAME_ITERATOR = 0
KEYS_IDS = list(range(49, 58))
KEYS_NAMES = [str(num) for num in range(1, 10)]
# add colors
color_list = list(itertools.combinations_with_replacement([0, 155, 255], 3))
random.shuffle(color_list)

# check files integrity
if not os.path.exists(DATAPOINTS_STRING):
    os.mkdir(DATAPOINTS_STRING)
if not os.path.exists(IMAGE_FILE_STRING):
    print("No database found, add images to ./database/Images directory")
    os.makedirs(IMAGE_FILE_STRING)
    os.makedirs(FILE_DONE_STRING)
    sys.exit()

# check output directory exist or make new Crops file
class_list = os.listdir(DATAPOINTS_STRING)
if len(class_list) == 0:
    os.mkdir(DATAPOINTS_STRING + 'Crops')
    print('NO CLASSES FOUND\n\nAdded Crops class')
    class_list = os.listdir(DATAPOINTS_STRING)
# check if classes extend 9 files
if len(class_list) > 9:
    print("Too many classes in datapoints directory\nMax number of classes in single run is 9")
    sys.exit()

# leave only relevant KEYS and colors
KEYS_IDS = KEYS_IDS[0:len(class_list)]
KEYS_NAMES = KEYS_NAMES[0:len(class_list)]
color_list = color_list[0: len(class_list)]

# make file for images done:
if not os.path.exists(FILE_DONE_STRING):
    os.mkdir(FILE_DONE_STRING)

# get paths from image folder:
imagePaths = list(paths.list_images(IMAGE_FILE_STRING))

for path in imagePaths:
    # load image
    image = cv2.imread(path)
    output_image = image.copy()
    # add instructions to image
    cv2.putText(image, "ESC to exit", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (200, 255, 0), 2)
    cv2.putText(image, "SPACE for next image", (0, 45), cv2.FONT_HERSHEY_PLAIN, 1.3, (200, 255, 0), 2)
    for i in range(0, len(class_list)):
        cv2.putText(image, "{} to save as {} class".format(KEYS_NAMES[i], class_list[i]), (0, 70 + i * 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, color_list[i], 2)

    # set initial image constraints and variables list
    h, w = image.shape[:2]
    x_max = w - screen_width - 1
    y_max = h - screen_height - 1
    pos_x = 0
    pos_y = 0
    point_list = []
    global_centers = []

    # add named window
    cv2.namedWindow(WINDOW_NAME)
    # add track bars
    cv2.createTrackbar(TRACKBAR_HORIZONTAL, WINDOW_NAME, pos_x, x_max, on_trackbar_horizontal)
    cv2.createTrackbar(TRACKBAR_VERTICAL, WINDOW_NAME, pos_y, y_max, on_trackbar_vertical)
    # add mouse callback
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        on_trackbar_horizontal(0)
        on_trackbar_vertical(0)
        key = cv2.waitKey(20)
        if key in KEYS_IDS:
            for point in global_centers:
                # get top left point of crop
                top_left_x = point[0] - HALF_SIZE
                top_left_y = point[1] - HALF_SIZE
                bottom_right_x = point[0] + HALF_SIZE
                bottom_right_y = point[1] + HALF_SIZE
                # check if top left is less than 0
                if top_left_x < 0:
                    top_left_x = 0
                    bottom_right_x = top_left_x + FRAME_SIZE
                if top_left_y < 0:
                    top_left_y = 0
                    bottom_right_y = top_left_y + FRAME_SIZE
                # check if bottom right is greater than image size
                if bottom_right_x >= w - 1:
                    top_left_x -= (bottom_right_x - w)
                    bottom_right_x = top_left_x + FRAME_SIZE
                if bottom_right_y >= h - 1:
                    top_left_y -= (bottom_right_y - h)
                    bottom_right_y = top_left_y + FRAME_SIZE

                # resize to 224x224 and save crop to file
                tile = output_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                tile = cv2.resize(tile, (224, 224), interpolation=cv2.INTER_CUBIC)

                for i in range(0, len(class_list)):
                    if key == KEYS_IDS[i]:
                        # check if file exists
                        output_path = DATAPOINTS_STRING + class_list[i] + \
                        os_separator + class_list[i] + "_" + \
                                      str(FRAME_ITERATOR) + "_s_{}".format(ZOOM_FACTOR).strip() + ".png"
                        while os.path.exists(output_path):
                            FRAME_ITERATOR += 1
                            output_path = DATAPOINTS_STRING + class_list[i] + \
                            os_separator + class_list[i] + "_" + \
                                          str(FRAME_ITERATOR) + "_s_{}".format(ZOOM_FACTOR).strip() + ".png"

                        # show crops on image
                        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color_list[i], 2)
                        cv2.imwrite(output_path, tile)

            # empty global centers and point list
            global_centers = []
            point_list = []

        if key == 32:
            pos_x = 0
            pos_y = 0
            # reset track bars
            cv2.setTrackbarPos(TRACKBAR_HORIZONTAL, WINDOW_NAME, 0)
            cv2.setTrackbarPos(TRACKBAR_VERTICAL, WINDOW_NAME, 0)
            # destroy window to initialize new trakc bars values
            cv2.destroyAllWindows()
            # move current image to DONE folder
            os.rename(path, path.replace('Images', 'DONE'))
            break

        # exit
        if key == 27:
            sys.exit()

print("All images done! Press ENTER to exit.")
