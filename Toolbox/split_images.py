#! /usr/bin/env python3

"""
Randomly splits a collection of images in a given folder into three sets: test,
train and validation with specified size.
"""

import os
import time
from random import sample
###############################################################################
accepted_extensions = [".jpg", ".png", ".ppm"]
print("Specify directory or leave empty to work in the current one:")
target_folder_with_images = input('>>  ')
print("Specify number of images for the 'training' folder:")
n_train = int(input('>>  '))
print("Specify number of images for the 'testing' folder:")
n_test = int(input('>>  '))
print("Specify number of images for the 'validation' folder:")
n_val = int(input('>>  '))
###############################################################################
if not target_folder_with_images:
    where = os.getcwd()
else:
    where = target_folder_with_images
print("Scanning in: " + where)
os.chdir(where)
fileNames = [f for f in os.listdir() for ext in accepted_extensions if f.endswith(ext)]
total = len(fileNames)
if not fileNames:
    print("No images found")
    raise SystemExit
else:
    print("Found {} image(s)".format(total))

train_set = sample(fileNames, k=n_train)
print("Training set: {}".format(train_set))
remainder1 = list(set(fileNames) - set(train_set))
test_set = sample(remainder1, k=n_test)
print("Testing set: {}".format(test_set))
remainder2 = list(set(remainder1) - set(test_set))
val_set = sample(remainder2, k=n_val)
print("Validation set: {}".format(val_set))

counter = 0
train_fol = 'training_set' + os.path.sep + target_folder_with_images
test_fol = 'testing_set' + os.path.sep + target_folder_with_images
val_fol = 'validation_set' + os.path.sep + target_folder_with_images
os.makedirs(train_fol, exist_ok=True)
os.makedirs(test_fol, exist_ok=True)
os.makedirs(val_fol, exist_ok=True)
for f in train_set:
    os.replace(f, train_fol + os.path.sep + str(f))
    counter += 1
for f in test_set:
    os.replace(f, test_fol + os.path.sep + str(f))
    counter += 1
for f in val_set:
    os.replace(f, val_fol + os.path.sep + str(f))
    counter += 1
print("Moved {} files".format(counter))
