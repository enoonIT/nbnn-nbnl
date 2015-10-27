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
    parser.add_argument("--patches", dest="patches", type=int, default=100,
                        help="Number of patches to extract per image.")

    args = parser.parse_args()

    if not args.input_dir:
        log.error('input-dir option is required, but not present.')
        exit()

    if not args.output_dir:
        log.error('output-dir option is required, but not present.')
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
        self.hfile = HDF5File(output_filename, 'w', compression='gzip', fillvalue=0.0)
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

def explode(key, patches, positions, output_dir, patchesPerImage):
    output_name = key + ".hdf5"
    ds = Dataset(output_name, output_dir, 1, patchesPerImage,
                 'DECAF', patch_dim=patches.shape[1],
                 patch_type='float32', pos_type='uint16')
    ds.append(key, patches, positions)
    get_logger().info("dataset with " + str(ds.keys.shape) + " elements, and patches " + str(patches.shape))
    ds.close()

if __name__ == '__main__':
    init_logging()
    log = get_logger()

    args = get_arguments()
    # Determining image files to extract from
    files = sorted(glob(join(args.input_dir, '*.hdf5') ), key=basename)
    start = time.time()

    for f in files:
        print "Loading file " + f
        dir_name = basename((splitext(f)[0]))
        hfile = HDF5File(f, 'r', compression='gzip', fillvalue=0.0)
        patches = hfile['patches']
        positions = hfile['positions']
        image_index = hfile['image_index']
        keys = hfile['keys']
        for k in range(len(keys)):
            outdir = join(args.output_dir, dir_name)
            iid = image_index[k]
            explode(keys[k], patches[iid[0]:iid[1]], positions[iid[0]:iid[1]], outdir, args.patches)
        hfile.close()
    end = time.time()
    print "It took " + str(end-start)
