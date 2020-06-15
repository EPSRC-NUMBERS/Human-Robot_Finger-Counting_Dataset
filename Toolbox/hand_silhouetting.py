#! /usr/bin/env python3

"""
Usage -> python3 hand_silhouetting.py /path/to/dataset/configExample.cfg
Make sure to place the config file in the root of the image dataset.

This script performs image segmentation to extract hand silhouettes.
Optionally, these can be combined with other backgrounds.
Resulting ROIs are saved in a separate file.
Original files are not changed.
"""

from pathlib import Path, PurePath
from random import choice
from argparse import ArgumentParser
import cv2
import numpy as np
from pandas import DataFrame

import libconf

import utils as fn

parser = ArgumentParser()
parser.add_argument("configFile")
args = parser.parse_args()

# Read directory structure
configPath = Path.resolve(Path.cwd() / args.configFile.strip("\""))
print(configPath)
basePath = configPath.parent
print("\nWorking directory: '{}'".format(basePath))

with open(configPath, encoding='utf-8') as f:
    print("Reading configuration file: '{}'".format(configPath))
    config = libconf.load(f)

print("IDs: '{}'".format(config.classes))

rF = Path.resolve(basePath / config.resultFolder)
rF.mkdir(mode=0o777, parents=True, exist_ok=True)
print("Resulting files will be saved in {}".format(rF))

# Read background images for combination
backgroundFolder = (basePath / config.newBackgrounds)
backgroundFiles = []
for e in config.acceptedExtensions:
    backgroundFiles.extend(list(backgroundFolder.glob('*.{}'.format(e))))
nBackFiles = len(backgroundFiles)
print("Number of background files: {}".format(nBackFiles))

# Read background image for threshold
imgBk = cv2.imread(str(basePath / config.classes[0] / config.backgroundFile), 1)

# Check mean value for background
if config.handType == "robot": # Convert BGR to HSV for robot hand
    imgBkConverted = cv2.cvtColor(imgBk, cv2.COLOR_BGR2HSV)
    # Detection of background value for threshold
    imgBk_ch = cv2.split(imgBkConverted)
    meanValueBK = cv2.mean(np.array(imgBk_ch[config.channel]))
    meanValueBK = meanValueBK[0]
    print("Mean Background Value: {}".format(meanValueBK))
else: # Convert BGR to YUV for human hand
    imgBkConverted = cv2.cvtColor(imgBk, cv2.COLOR_BGR2YUV)

# Number words dictionary
NW = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five'}

allData = []
counter = 0
for item in config.classes[1:]:
    currentFolder = basePath / item
    print("Reading folder: '{}'".format(currentFolder))
    imageNumber = 0
    fileNameList = []
    for e in config.acceptedExtensions:
        fileNameList.extend(list(currentFolder.glob('*.{}'.format(e))))
    for fileName in fileNameList:
        imageName = currentFolder.joinpath(fileName)

        img = cv2.imread(str(imageName), 1)
        print("Processing image: {}".format(imageName))

        if config.handType == "robot": # HSV works better for robot hands
            imgConvert = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_split = cv2.split(imgConvert)
            mask = cv2.inRange(img_split[config.channel],
                                meanValueBK-config.threshold,
                                meanValueBK+config.threshold)
            mask = cv2.bitwise_not(mask)
            cv2.imshow('Mask', mask)
            k = cv2.waitKey(config.waitTime)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.dilate(mask, kernel, iterations = 1)
            mask = cv2.erode(mask, kernel, iterations = 2)
            object = mask
            object_= object
        else: # YUV works better for robot hands
            # Object detection with threshold
            imgConvert = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            object, mask = fn.detectObject(imgBkConverted, imgConvert, config.threshold)

        # Detect contours of all objects in the image
        contours, hierarchy = cv2.findContours(object,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if not object.any() or not contours:
            print("No contours identified on this image: '{}'".format(imageName))
            continue
        # Detect the bigger object and obtains its bounding box
        x, y, w, h, index = fn.boundingBox(contours)
        # Create a mask of the bigger object to remove noise. The original one has noise because of the light
        maskObject = np.zeros((object.shape), np.uint8)
        cv2.drawContours(maskObject, [contours[index]], -1, (255,255,255), -1, cv2.LINE_AA)
        # Defines the set of points of the contour. The original ones and the filtered ones
        contoursPoints = cv2.approxPolyDP(contours[index], config.approximation, 1)
        contoursFiltered = contoursPoints

        # Create the new image. The new image is a combination of the original and a background image.
        # If backgrounds folder is empty, combination is not done
        if nBackFiles > 0:
            imgBkFileName = choice(backgroundFiles)
            imgBackground = cv2.imread(str(imgBkFileName), 1)
            imgComb = fn.combineImages(img, imgBackground, mask)
        else:
            imgComb = img

        # Generate file name
        fileNameSave_ = '{}_{}.jpg'.format(NW[item], imageNumber)

        # Extract features of the object
        data = fn.features(img, maskObject, contours[index])
        imgInfo = [fileNameSave_, item]
        imgInfo.extend(data)
        allData.append(imgInfo)

        # Save results in an XML file to use it with TensorFlow
        height, width, channels = img.shape
        fn.saveFileXML(fileNameSave_, config.resultFolder, basePath, height, width, channels, item, x, y, w, h);

        # Save the image
        fileNameSave = Path.resolve(basePath / config.resultFolder / fileNameSave_)
        cv2.imwrite(str(fileNameSave), imgComb)

        # Save the mask
        fileNameSaveMask = Path(PurePath(fileNameSave).with_suffix(".png")).resolve()
        cv2.imwrite(str(fileNameSaveMask), maskObject)

        # Show results
        boundingBox = [x, y, x+w, y+h]
        imgFeatures = fn.drawFeatures(imgComb, contoursPoints, contoursFiltered, boundingBox, imgInfo)
        cv2.putText(mask,"MASK",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(object,"OBJECT",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(maskObject,"MASK_OBJECT",(25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        mask_ = cv2.resize(mask,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        object_ = cv2.resize(object,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        maskObject_ = cv2.resize(maskObject,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        numpy_vertical_1 = np.vstack((object_, maskObject_))
        numpy_vertical_1 = np.vstack((numpy_vertical_1, mask_))

        imgR = cv2.resize(imgConvert,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        imgR_S = cv2.split(imgR)
        numpy_vertical_2 = np.vstack((imgR_S[0], imgR_S[1]))
        numpy_vertical_2 = np.vstack((numpy_vertical_2, imgR_S[2]))

        cv2.imshow('Masks', numpy_vertical_1)
        cv2.imshow('HSV(robot)/YUV(human)', numpy_vertical_2)
        cv2.imshow('Image', imgComb)
        k = cv2.waitKey(config.waitTime)

        # Iterate
        print("--> {}".format(fileNameSave))
        imageNumber += 1
        counter += 1

column_name = [
        'filename', 'class', 'cx', 'cy', 'area', 'perimeter', 'angle',
        'aspectRatio', 'solidity', 'cy_iluWeight', 'cx_iluWeight', 'mean_Blue',
        'mean_Green', 'mean_Red', 'minorAxis', 'majorAxis', 'xEllipse', 'yEllipse',
        'angleIllumination', 'distanceIllumination'
    ]
allData_df = DataFrame(allData, columns=column_name)
path_to_allData = Path.resolve(basePath / config.resultFolder / config.dataFile)
allData_df.to_csv(str(path_to_allData), index=None)
print("All data saved to .csv file: '{}'".format(path_to_allData))
print("Processed {} images.".format(counter))

cv2.destroyAllWindows()

fn.xml_to_csv(Path.resolve(basePath / config.resultFolder))
