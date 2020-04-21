

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3755452.svg)](https://doi.org/10.5281/zenodo.3755452)


# KrakN 

<p align="center">
  <img src="https://i.imgur.com/DPoNWEX.png" alt="KrakN project logo" width="600" height="330"/>
</p>

>**KrakN is an Artificial Intelligence framework for supporting infrastructure maintenance. It provides tools for an easy, end-to-end approach to dataset building, training, and deploying CNN models for image defect detecting on infrastructure facilities for infrastructure management units. KrakN is free to use, and it is distributed under open source license.**


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)
* [Use examples](#use-examples)
* [Acknowledgments](#acknowledgments)

## General info
KrakN project was created to provide a comprehensive and reliable software package supporting infrastructure inspections with Artificial Intelligence methods. It is designed to improve the accuracy of defect detection of the Bridge Inspectors by providing them with tools to train and use CNN classifiers that will match their work-specific problems.

With the use of KrakN it is possible to achieve over 95% of accuracy in detecting defects as small as 0.2 mm wide cracks.

KrakN consists of two main modules:
* Dataset builder
* Network trainer and Defect detector

Dataset builder enables semin-automatic construction of own datasets based on photos taken during the infrastructure inspection. With its help, based on high-resolution images, it is possible to create large datasets for training convolutional neural networks.

Defect Detector allows for training and deploying CNN classifiers on digital pictures in real-life working scenarios. It outputs immediate defect indicators for use during inspection as well as defect masks for further defect management when used on surface orthomosaic.

KrakN can be used on desktop as well as in cloud. It comes in three variants:
* Windows / Linux for dekstop use
* Precompiled Windows version
* Google Colab for cloud computing

## Dependencies

KrakeN was developed using Python 3.6. It can be used on Microsoft Windows as well as on most GNU/Linux distributions. It requires the following packages
* TensorFlow == 1.12.0
* Keras >= 2.2.4
* Scikit-learn >= 0.22.1
* Numpy == 1.16.2
* OpenCV >= 4.0.0
* H5Py >= 2.9.0
* Progressbar >= 2.5
* Imutils >= 0.5.2
* Pillow >= 7.1.1
* PyGame >= 1.9.6

### Setup


#### Using `pip`

In order to setup required dependencies, run `chmod +x ./install_dependencies.sh & ./install_dependencies.sh` in KrakN directory for Linux or `install_dependencies.bat` in CMD for Windows. 

You can also install required libraries manually with `pip3 install <module_name> == <module_version>` as all of required libraries come in `pip` packages.

GPU acceleration is also available with `tensorflow-gpu` and CUDA installed. For further information see [CUDA install guide](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781).

KrakN comes also in pre-compiled `.exe` files for Windows that do not require any installation, however `.exe` distribution does not support GPU-CUDA acceleration.

#### Using Anaconda

If you are using [Anaconda](https://www.anaconda.com/), you can use `krank.yml` file to create environment with all dependencies required to use tha package.

```
conda env create -f krakn.yml
```

#### Using precompiled `.exe` files

If you are a Windows user, you can download precompiled executable files from [Zenodo software repository](https://www.zenodo.org/record/3759845) and run them on your machine without any further installation (although you should keep the directory structure - KrakN will do it for you by adding folders in first run, but then again, first run would be disappointing).

As KrakN uses command line as a form of communication with user, the preferred way of executing software is by typing eg. `dataset_builder.exe` in CMD or `.\dataset_builder.exe` in PowerShell opened in the `*.exe` directory. This way you will be able to read any error messages if they occur. However it is also possible to run KrakN software with simple double click.

To further simplify the use of the software, the sections below are provided with icon images corresponding to individual software components.

## Usage

### 1. Dataset Building
<img src="https://i.imgur.com/sKTRyBu.png" alt="drawing" width="200" height="150"/>

In order to build dataset, first create directories with names corresponding to desired classifier classes in `KrakN/dataset_builder/datapoints`. Adding each new folder is equivalent in adding new class to the desired classifier. There can be total of 9 folders in `KrakN/dataset_builder/datapoints` per script run. Next add images you want to extract dataset from to `KrakN/dataset_builder/database/Images`. Finally, use `dataset_builder.py` or `dataset_builder.exe` in `KrakN/dataset_builder` to extract labeled datapoints from images.

#### Datapoints extraction:

After running `dataset_builder` you will be prompted to enter zoom factor for managing output datapoint size. Set zoom factor to match the defect size.

Use mouse to select paths of the datapoints extraction. Each pair of mouse clicks will add another line to path:

![](https://i.ibb.co/KVgqdmF/1s.png)
>By default, one path allows to extract 5 datapoints.

Then use controls to extract crops out of image:

![](https://i.ibb.co/LgwgPk9/Zrzut-ekranu-z-2020-03-26-21-03-11.png)
>Controls are listed in top-left corner of the image window.

Extracted datapoints will be saved in their corresponding directories:

![](https://i.ibb.co/pKgW3xX/2s.png)
>Each of the extracted datapoint will be marked with rectangle

Use trackbars to navigate image.

![](https://i.ibb.co/ZTL1H7H/3s.png)

There is no limitation to the number of datapoints extracted from one image.

Dataset classes should be balanced and consist of at least 500 datapoints per class.

### 2. Feature extraction
<img src="https://i.imgur.com/YusYHOK.png" alt="drawing" width="200" height="150"/>

In order to train your classifier, first you have to extract image features from your gathered dataset. Use `extract_features.py` or `extract_features.exe` in `KrakN/network_trainer` to begin extraction.

You will have to provide path to your dataset in case if multiple dataset directories are present.

With first use of the feature extractor, a VGG16 headless CNN model trained on ImageNet dataset will be downloaded to your machine. The model size is ~55MB. 

This process can also be done with `extract_features.py` provided in `KrakN/network_trainer_Colab` for greater performance with use of [Google Colab](https://colab.research.google.com) cloud computing service. However, in order to do that, your dataset has to be present on your Google Drive.

After feature extraction, your dataset will be saved in `KrakN/network_trainer/database` as `features_s_<zoom factor>.hdf5` file.

**For fine crack/background detection database, already extracted features can be downloaded from [project's Zenodo site](https://zenodo.org/record/3755452).**

### 3. Network training
<img src="https://i.imgur.com/hAUwoN5.png" alt="drawing" width="200" height="150"/>

In order to train network run `train_model.py` or `train_model.exe`. It will automatically load your dataset if provided correctly and tune training hyperparameters. Total training time should not exceed 1 hour and the trained classifier will be saved as `KrakN_model.cpickle` in the `KrakN/network_trainer` directory.

**After training the model, points 1 to 3 will not have to be repeated unless you will need to add new classes to the classifier.** Trained models can also be shared across working stations.

### 4. Model deployment

<img src="https://i.imgur.com/xV2GcHe.png" alt="drawing" width="200" height="170"/>

#### Single stage classification

In order to use your trained model, first insert images you want to classify to `KrakN/network_trainer/input` directory. Then use `moving_window.py` or `moving_window.exe` in `KrakN/network_trainer` directory to classify images.

KrakN will automatically load your images, database and trained model. It will then classify your images and output results to the `KrakN/network_trainer/output` directory.

There are no limitations to the size of the images classified, so images as large as 120MP orthomosaic can be classified.

For greater performance, Google Colab service can also be used.

#### Subsequent stage classification

KrakN also allows for multi stage classification, with `network_trainer_Staged` module.

In order to do so, you have to train new classifiers using previously described methods. The classifiers should be renamed to match first stage of defect detection - eg. if second stage classification is meant to divide 'spalling' defect into subsets, classifier should be named 'spalling' accordingly as seen in the figure below. Next, place classifiers in `network_trainer_Staged/Classifiers` directory and insert previously classified images and masks to corresponding folders. Note, that image and classifier names should match. To classify the images, run `moving_window_staged.py`. You can use any number of images and classifiers at once. As the result, classified images will be saved in `network_trainer_Staged/Output` both as images and masks. Multi stage classification can be done repeatedly.

<img src="https://i.imgur.com/ngLwPfm.jpg" alt="drawing" width="450" height="240"/>

## Use examples

#### Examples below demonstrate classifications obtained using KrakN:

<img src="https://i.ibb.co/8bL6B9G/out1.png" alt="drawing" width="640" height="230"/>

>A fragment of KrakN output of concrete wall 120MP orthomosaic. Second image in row is the defect mask to use for damage management. On the last image, the defects are marked with bounding boxes. The width of defects is below 0,2mm.

<img src="https://i.ibb.co/NZcxLY5/out2.png" alt="drawing" width="640" height="230"/>

>Another example of KrakN use - note false positive marking in the top right corner.

<img src="https://i.ibb.co/VvpxLs4/out3.png" alt="drawing" width="640" height="230"/>

>An example of KrakN usage on clean, painted surface it was not trained on - note false negative predictions on the image.

As seen in the examples KrakN performance may vary due to the dataset it was trained on. To maintain high performance in defect detection, use diverse datasets for training. Further fine tunning in KrakN performance can be done by manipulating `confidence_threshold` parameter in `moving_window.py` for lowering false negative predictions.

## Acknowledgments

KrakN is associated with research paper:

Mateusz Żarski<sup>[1](#footnote1)</sup>, Bartosz Wójcik<sup>[1](#footnote1)</sup>, Jarosław Adam Miszczak<sup>[2](#footnote2)</sup>, *Transfer Learning for leveraging the utilization of computer vision for infrastructure maintenance*, [ResearchGate link](https://researchgate.net) 

<a name="footnote1"><sup>1</sup></a>Department of Civil Engineering, Silesian University of Technology
<a name="footnote2"><sup>2</sup></a>Institute of Theoretical and Applied Informatics, Polish Academy of Sciences

KrakN is free to use under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

*This project was supported by Polish National Center for Research and Developement
under grant number POWR.03.05.00-00.z098/17-00.*
