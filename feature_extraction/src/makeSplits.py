#!/usr/bin/python

########################################
# Patch/descriptor extraction utility. #
#                                      #
# Author: fabiom.carlucci@yahoo.it     #
########################################

from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename, exists
from os import makedirs, remove
from common import init_logging, get_logger
import random
import math
import time
from itertools import product

import numpy as np
from h5py import File as HDF5File
from h5py import special_dtype

from common import get_desc_name

def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='HD5 Splitter.')
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Directory with HDF5 images.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory to put HDF5 files to.")
    parser.add_argument("--num-splits", dest="num_splits", type=int,
                        help="Number of splits to make.")
    parser.add_argument("--num-train-images", dest="num_train_images", type=int,
                        help="Number of train images.")
    parser.add_argument("--num-test-images", dest="num_test_images", type=int,
                        help="Number of test images.")
    parser.add_argument("--patches", dest="patches", type=int, default=100,
                        help="Number of patches to extract per image.")
    parser.add_argument('--relu', dest='relu', action='store_true', default=False)
    parser.add_argument('--limitCategories', dest='limit', action='store_true', default=False)
    parser.add_argument("--category_list", dest="categories",
                        help="File containing list of valid categories")
    
    args = parser.parse_args()

    if not args.input_dir:
        log.error('input-dir option is required, but not present.')
        exit()

    if not args.output_dir:
        log.error('output-dir option is required, but not present.')
        exit()

    if not args.num_splits:
        log.error('num-splits option is required, but not present.')
        exit()

    if not args.num_train_images:
        log.error('num_train_images option is required, but not present.')
        exit()

    if not args.num_test_images:
        log.error('num_test_images option is required, but not present.')
        exit()
    return args


class Dataset:
##################################################################################
# Class responsible for storing descriptors and their metadata to the HDF5 file. #
# The process of storing is incremental by calling append().                     #
##################################################################################

    def __init__(self, output_name, output_dir, num_files, patches, feature_type,
                 patch_dim=128, patch_type='uint8', pos_type='uint16'):
        self.log = get_logger()

        output_subdir = output_dir
        try:
            makedirs(output_subdir)
        except:
            pass

        output_filename = join(output_subdir, basename(output_name))
        self.log.debug('Saving extracted descriptors to %s', output_filename)

        self.mode = 'creating'
        dt = special_dtype(vlen=bytes)
        patches += 10 #for safety
        self.hfile = HDF5File(output_filename, 'w', compression='gzip', compression_opts=9, fillvalue=0.0)
        self.patches = self.hfile.create_dataset('patches', (num_files * patches, patch_dim), dtype=patch_type, chunks=True)
        self.positions = self.hfile.create_dataset('positions', (num_files * patches, 2), dtype=pos_type, chunks=True)
        self.image_index = self.hfile.create_dataset('image_index', (num_files, 2), dtype='uint64') # Start, End positions of an image
        self.keys = self.hfile.create_dataset('keys', (num_files, ), dtype=dt)
        self.key_set = set()
        self.patches.attrs['cursor'] = 0
        self.patches.attrs['feature_type'] = feature_type

        self.output_filename = output_filename

    def __exit__(self, type, value, traceback):
        self.hfile.close()

    def __contains__(self, key):
        return key in self.key_set

    def append(self, key, patches, pos):
        num_patches = patches.shape[0]
        num_keys = len(self.key_set)
        assert(num_patches == pos.shape[0])

        start = self.patches.attrs['cursor']
        end = self.patches.attrs['cursor'] + num_patches
        self.patches[start:end, :] = patches
        self.positions[start:end, :] = pos
        self.image_index[num_keys, 0] = start
        self.image_index[num_keys, 1] = end
        self.keys[num_keys] = key
        self.key_set.add(key)
        self.patches.attrs['cursor'] += num_patches

    def close(self):
        self.hfile.close()

def make_split(imageNumbers, output_name, output_dir, patches, positions, image_indexes, keys, patchesPerImage, relu):
    ds = Dataset(output_name, output_dir, len(imageNumbers),
                 patchesPerImage,
                 'DECAF', patch_dim=patches.shape[1],
                 patch_type='float', pos_type='uint16')
    for idx in imageNumbers:
        ir = image_indexes[idx]
        p = patches[ir[0]:ir[1]]
        if(relu):
            negs = p < 0
            p[negs] = 0
        ds.append(keys[idx], p, positions[ir[0]:ir[1]])
    print "dataset with " + str(ds.keys.shape) + " elements"
    ds.close()
    
def get_legal_categories(categories_file):
    with open(categories_file) as f:
        lines = f.read().splitlines()
        return lines
    
    
if __name__ == '__main__':
    init_logging()
    log = get_logger()

    args = get_arguments()
    if args.relu:
      print "RELU active"
    else:
      print "RELU not active"
    if args.limit:
        legalCategories = get_legal_categories()
        print "Using subset of categories"
    # Determining image files to extract from
    files = sorted(glob(join(args.input_dir, '*.hdf5') ), key=basename)
    start = time.time()
    
    for f in files:
        print "Loading file " + f
        if(args.limit and not(f in legalCategories)):
            print f + " not in selected categories, skipping"
            continue
        hfile = HDF5File(f, 'r', compression='gzip', compression_opts=9, fillvalue=0.0)
        patches = hfile['patches']
        positions = hfile['positions']
        image_index = hfile['image_index']
        keys = hfile['keys']
        try:
            nPatches = patches.attrs['n_patches']
        except:
            nPatches = 100
        print "Patches " + str(patches.shape) + " positions: " + str(positions.shape) + " Image index: " + str(image_index.shape) + " keys: " + str(keys.shape)
        imageList = range(keys.shape[0]) #makes a list 0 to keys.shape[0]-1
        for x in range(1,args.num_splits+1):
            print "Starting split " + str(x)
            random.shuffle(imageList)
            trainIndexes = imageList[0:args.num_train_images]
            if(args.num_test_images==-1):
                testIndexes = imageList[args.num_train_images:]
            else:
                testIndexes = imageList[args.num_train_images:args.num_train_images+args.num_test_images]
            make_split(trainIndexes, f, join(args.output_dir, 'train', 'split_%d' % x), patches, positions, image_index, keys, nPatches, args.relu)
            make_split(testIndexes, f, join(args.output_dir, 'test', 'split_%d' % x), patches, positions, image_index, keys, nPatches, args.relu)
        hfile.close()
    end = time.time()
    print "It took " + str(end-start)
