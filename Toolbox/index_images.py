#! /usr/bin/env python3

"""
Rename images in a given folder based on their last modified time in a
numerical order
"""

import os
import time
###############################################################################
accepted_extensions = [".jpg", ".png", ".ppm"]

print("Specify directory or leave empty to work in the current one:")
target_folder_with_images = input('>>  ')
print("Specify prefix:")
prefix = input('>>  ')
###############################################################################
if not target_folder_with_images:
    where = os.getcwd()
else:
    where = target_folder_with_images
print("Scanning in: " + where)
os.chdir(where)
counter = 0
for ext in accepted_extensions:
    candidates = [(im, os.path.getmtime(im), str(hash(im+str(time.time()))) + ext) \
        for im in os.listdir() if ext in os.path.splitext(im)[1]]
    # Sort by 'last modified' time or by name if equal
    candidates.sort(key=lambda tup: (tup[1], tup [0]))
    for img in candidates:
        os.rename(img[0], img[2])
    for img in candidates:
        counter += 1
        new_name = prefix + str(counter) + ext
        os.rename(img[2], new_name)
        print('Renamed {} to {}'.format(img[0], new_name))

print("Renamed {} files".format(counter))
