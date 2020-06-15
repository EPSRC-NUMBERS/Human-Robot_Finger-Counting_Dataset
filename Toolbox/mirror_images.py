#! /usr/bin/env python3

"""
Flip images horizontally and/or vertically in a given folder
"""

import os
import cv2
###############################################################################
accepted_extensions = [".jpg", ".png", ".ppm"]
vertical_flip = False # along x-axis
horizontal_flip = True # along y-axis
remove_originals = True

print("Specify directory or leave empty to work in the current one:")
target_folder_with_images = input('>>  ')
###############################################################################
if not target_folder_with_images:
    where = os.getcwd()
else:
    where = target_folder_with_images
fileNameList = [fn for fn in os.listdir(where) if os.path.splitext(fn)[1] in \
                    accepted_extensions]
for fn in fileNameList:
    img = cv2.imread(os.path.join(where,fn))
    deep_copy = img.copy()
    if vertical_flip:
        deep_copy = cv2.flip(deep_copy, 0)
    if horizontal_flip:
        deep_copy = cv2.flip(deep_copy, 1)
    new_name = os.path.splitext(fn)[0] + '_mirrored' + os.path.splitext(fn)[1]
    cv2.imwrite(os.path.join(where, new_name), deep_copy)
    if remove_originals:
        try:
            os.remove(os.path.join(where,fn))
        except Exception as e:
            print(e)
