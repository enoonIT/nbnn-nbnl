from glob import glob
import numpy as np
from h5py import File as hfile
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser(description='Utility to load training\testing splits '
    'for the Washington RGB-D Object Dataset.')
    parser.add_argument("mode", help="What should this script do. 'gen' or 'svm'")
    parser.add_argument("input_folder", help="The folder containing the extracted features")
    parser.add_argument("--output_folder", help="The folder where to position the split files")
    parser.add_argument("split_file", help="The file from which to load the splits")
    parser.add_argument("split", help="Which training split to load", type=int)
    parser.add_argument("--prefix", help="Optional prefix for train\test.txt files")
    parser.add_argument("--rnd_split", help="Make random splits, need n_train and n_test")
    parser.add_argument("--n_train", help="Number of training images", type=int)
    parser.add_argument("--n_test", help="Number of testing images", type=int)
    args = parser.parse_args()
    return args

