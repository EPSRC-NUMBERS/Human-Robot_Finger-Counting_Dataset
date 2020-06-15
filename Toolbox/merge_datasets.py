#! /usr/bin/env python3

"""
Usage -> python3 merge_datasets.py --dataset1 path/to/dataset1
--dataset2 path/to/dataset2 --output path/to/output/folder

Merge two datasets: including images, masks and labels, together with
the aggregated metadata (.csv files).
"""

import os
from argparse import ArgumentParser
from shutil import copy2
import pathlib as pb
import cv2
import pandas as pd

parser = ArgumentParser(description= 'Merge two datasets')
parser.add_argument("-D1", "--dataset1", type=str,
    help="Path to the first dataset")
parser.add_argument("-D2", "--dataset2", type=str,
    help="Path to the second dataset")
parser.add_argument("-O", "--output", type=str,
    help="Path to resulting dataset, will be created")
parser.add_argument("-AL", "--all_labels", type=str,
    help="Name of the file with labels",
    default="all_labels.csv")
parser.add_argument("-AD", "--all_data", type=str,
    help="Name of the file with metadata",
    default="allData.csv")


args = parser.parse_args()

allDataN = args.all_data
all_labelsN = args.all_labels

def check_files(pathlike):
    print("Checking for required files...")
    allData_file = (pathlike / allDataN)
    if allData_file.is_file():
        if allData_file.stat().st_size != 0:
            print("Found {}".format(allDataN))
        else:
            print("{} is empty".format(allDataN))
    else:
        print("Missing {}".format(allDataN))
    all_labels_file = (pathlike / all_labelsN)
    if all_labels_file.is_file():
        if all_labels_file.stat().st_size != 0:
            print("Found {}".format(all_labelsN))
        else:
            print("{} is empty".format(all_labelsN))
    else:
        print("Missing {}".format(all_labelsN))
    jpg_files = [f for f in pathlike.glob('*.jpg')]
    print("Found {} '.jpg' files".format(len(jpg_files)))
    png_files = [f for f in pathlike.glob('*.png')]
    print("Found {} '.png' files".format(len(png_files)))
    xml_files = [f for f in pathlike.glob('*.xml')]
    print("Found {} '.xml' files".format(len(xml_files)))


def main():
    # Dealing with Paths
    cwd = pb.Path.cwd()
    dataset1_pathlike = pb.Path(args.dataset1)
    dataset1_pathlike_agnostic = (cwd / dataset1_pathlike).resolve()
    print("Dataset1 directory: '{}'".format(dataset1_pathlike_agnostic))
    check_files(dataset1_pathlike_agnostic)

    dataset2_pathlike = pb.Path(args.dataset2)
    dataset2_pathlike_agnostic = (cwd / dataset2_pathlike).resolve()
    print("Dataset2 directory: '{}'".format(dataset2_pathlike_agnostic))
    check_files(dataset2_pathlike_agnostic)

    output_pathlike = pb.Path(args.output)
    output_pathlike_agnostic = (cwd / output_pathlike).resolve()
    out = output_pathlike_agnostic
    print("Output directory: '{}'".format(out))
    if out.is_dir():
        if os.listdir(out):
            print("Directory exists and is non-empty")
            raise SystemExit
        else:
            print("Directory exists and is empty")
    else:
        try:
            print("Creating output directory...")
            out.mkdir(mode=0o777, parents=True, exist_ok=True)
        except Exception as e:
            print(e)
        else:
            print("Output directory created")

    # Combine files

    dirs = [dataset1_pathlike_agnostic, dataset2_pathlike_agnostic]
    fileCounter = 0
    errorsN = 0
    # Create the new .csv files
    # Read in corresponding 'all_labels.csv' files
    all_labels_files = [pd.read_csv(d / all_labelsN) for d in dirs]
    # Read in corresponding 'allData.csv' files
    allData_files = [pd.read_csv(d / allDataN) for d in dirs]
    if list(all_labels_files[0]) != list(all_labels_files[1]):
        print('all_labels.csv dataset files are not of the same structure!')
        raise SystemExit
    else:
        all_labels_ = pd.DataFrame(columns=list(all_labels_files[0]))
        l_row_list = []

    if list(allData_files[0]) != list(allData_files[1]):
        print('allData.csv dataset files are not of the same structure!')
        raise SystemExit
    else:
        allData_ = pd.DataFrame(columns=list(allData_files[0]))
        d_row_list = []

    for ind, d in enumerate(dirs):
        l_df = all_labels_files[ind]
        d_df = allData_files[ind]
        # Read in images one by one
        jpg_files = [f for f in d.glob('*.jpg')]
        for f in jpg_files:
            fileCounter +=1
            print("Processing image {}: '{}'".format(fileCounter, f))
            xml_ = f.with_suffix('.xml')
            if not xml_.is_file():
                print('Skipping, as no corresponding .xml file was found: {}'.format(xml_))
                errorsN += 1
                continue
            png_ = f.with_suffix('.png')
            if not png_.is_file():
                print('Skipping, as no corresponding .png file was found: {}'.format(png_))
                errorsN += 1
                continue

            # Create new file names
            oldstem = f.stem
            oldname = f.name
            new_prefix = '{}_{}'.format(oldstem[:oldstem.rfind('_')], fileCounter)
            new_jpg = new_prefix + '.jpg'
            new_jpg_name = out / new_jpg
            new_png_name = out / (new_prefix + '.png')
            new_xml_name = out / (new_prefix + '.xml')

            # Copy data from corresponding 'all_labels.csv' file
            l_entry = l_df[l_df['filename'].str.match(oldname)].copy(deep=True)
            l_entry_dict = l_entry.to_dict(orient = 'list')
            l_entry_dict['filename'][0] = new_jpg
            l_entry_dict_flat = {ki:vi[0] for ki,vi in l_entry_dict.items()}
            l_row_list.append(l_entry_dict_flat)

            # Copy data from corresponding 'allData.csv' file
            d_entry = d_df[d_df['filename'].str.match(oldname)].copy(deep=True)
            d_entry_dict = d_entry.to_dict(orient = 'list')
            d_entry_dict['filename'][0] = new_jpg
            d_entry_dict_flat = {ki:vi[0] for ki,vi in d_entry_dict.items()}
            d_row_list.append(d_entry_dict_flat)

            # Making file copies
            copy2(f, new_jpg_name)
            print("Copying .jpg: '{}'".format(new_jpg_name))
            copy2(xml_, new_png_name)
            print("Copying .png: '{}'".format(new_png_name))
            copy2(png_, new_xml_name)
            print("Copying .xml: '{}'".format(new_xml_name))

    print('Writing combined labels to: {}'.format(out / all_labelsN))
    all_labels_ = pd.DataFrame.from_dict(l_row_list)
    all_labels_.to_csv(out / all_labelsN, index=False, columns=list(all_labels_files[0]))
    print('Writing combined data to: {}'.format(out / all_labelsN))
    allData_ = pd.DataFrame.from_dict(d_row_list)
    allData_.to_csv(out / allDataN, index=False, columns=list(allData_files[0]))

    if errorsN == 0:
        print('All {} files successfully copied!'.format(fileCounter))
    else:
        print('There were errors. {} files were copied, \
        and {} files were skipped'.format(fileCounter, errorsN))


if __name__ == '__main__':
    main()
