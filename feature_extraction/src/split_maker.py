# -*- coding: utf-8 -*-
'''This file contains methods to create split files starting from files containing the complete list of files available
    for a given dataset. It has the following methods:
        random split: needs to provide number of train and test images (todo)
        instance based split: reads a file to know which images are test and which are train
'''
from argparse import ArgumentParser
import re
nregex = re.compile("(\w+)\_\d")


def get_arguments():
    parser = ArgumentParser(description='Utility to load training\testing splits '
    'for the Washington RGB-D Object Dataset.')
    parser.add_argument("all_files", help="File containing the list of all images")
    #parser.add_argument("depth_file", help="File containing the list of ")
    parser.add_argument("split_file", help="The file from which to load the splits")
    parser.add_argument("output_folder", help="The folder where to position the split files")
    args = parser.parse_args()
    return args


def get_testing_instances(split_file):
    """Returns a list of lists containing the object instances we want to keep for testing"""
    instances = []
    with open(split_file) as sfile:
        data = sfile.read()
        trials = re.split("\*+\strial\s\d+\s\*+", data)[1:]  # first one is empty
        for trial in trials:
            instances.append(filter(None, trial.splitlines()))
        print("There are " + str(len(instances)) + " splits")
    return instances


def is_test_sample(sample, split_instances):
    instance = sample.split("/")[1]
    return instance in split_instances


def create_split_files(all_samples, all_split_instances, out_folder):
    '''all samples is a list of all the images in the dataset
    split_instances is a list containing the test instances for each split'''
    n_splits = len(all_split_instances)
    out_files_test = []
    out_files_train = []
    for x in range(n_splits):
        out_files_test.append(open(out_folder + "/depth_test_split_" + str(x) + ".txt", "w"))
        out_files_train.append(open(out_folder + "/depth_train_split_" + str(x) + ".txt", "w"))
    for sample in all_samples:
        for idx, split_instances in enumerate(all_split_instances):
            if is_test_sample(sample, split_instances):
                out_files_test[idx].write(sample)
            else:
                out_files_train[idx].write(sample)
    for x in range(n_splits):
        out_files_train[x].close()
        out_files_test[x].close()


def load_all_samples(file_path):
    data = []
    with open(file_path, "r") as sfile:
        data = sfile.readlines()
    return data


if __name__ == '__main__':
    args = get_arguments()
    all_split_instances = get_testing_instances(args.split_file)
    create_split_files(load_all_samples(args.all_files), all_split_instances, args.output_folder)
