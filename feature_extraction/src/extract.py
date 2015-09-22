#!/usr/bin/python

########################################
# Patch/descriptor extraction utility. #
#                                      #
# Author: ilja.kuzborskij@idiap.ch     #
########################################

from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename, exists
from os import makedirs, remove
from common import init_logging, get_logger
import random
import math
from itertools import product

import Image
import numpy as np
from h5py import File as HDF5File
from h5py import special_dtype

from common import get_desc_name
import CaffeExtractor

def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='Patch/descriptor extraction utility.')
    parser.add_argument("--patches", dest="patches", type=int, default=1000,
                        help="Number of patches to extract per image.")
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=16,
                        help="Size of the patch.")
    parser.add_argument("--image-dim", dest="image_dim", type=int,
                        help="Size of the largest image dimension.")
    parser.add_argument("--levels", dest="levels", type=int, default=3,
                        help="Number of hierarchical levels to extract patches from. Procedure starts from <patch-size> and divides it by 2 at each level.")
    parser.add_argument("--descriptor", dest="descriptor", default='DECAF',
                        choices=['DECAF'],
                        help="Type of feature descriptor.")
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Directory with JPEG images.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory to put HDF5 files to.")
    parser.add_argument("--num-train-images", dest="num_train_images", type=int,
                        help="Number of train images.")
    parser.add_argument("--num-test-images", dest="num_test_images", type=int,
                        help="Number of test images.")
    parser.add_argument("--split", dest="split", type=int,
                        help="Split to extract.")
    parser.add_argument("--oversample", dest="oversample", action='store_true',
                        help="Add patch rotations.")
    parser.add_argument("--decaf-oversample", dest="decaf_oversample", action='store_true',
                        help="DECAF oversampling. Flip/corner etc.")
    parser.add_argument("--layer-name", dest="layer_name",
                        help="Decaf layer name.")
    parser.add_argument("--network-data-dir", dest="network_data_dir",
                        help="Directory holding the network weights.")
    parser.add_argument("--patch-method", dest="patch_method",
                        help="What method to use to extract patches.")

    args = parser.parse_args()

    if not args.input_dir:
        log.error('input-dir option is required, but not present.')
        exit()

    if not args.output_dir:
        log.error('output-dir option is required, but not present.')
        exit()

    if not args.image_dim:
        log.error('image-dim option is required, but not present.')
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

    def __init__(self, input_dir, output_dir, num_files, patches, feature_type,
                 patch_dim=128, patch_type='uint8', pos_type='uint16'):
        self.log = get_logger()

        output_subdir = output_dir
        try:
            makedirs(output_subdir)
        except:
            pass

        output_filename = join(output_subdir, basename(input_dir.strip('/')) + '.hdf5')

        self.log.debug('Saving extracted descriptors to %s', output_filename)
        if exists(output_filename):
            self.mode = 'appending'

            self.log.warn('File "%s" already exists. Trying to continue.', output_filename)
            self.hfile = HDF5File(output_filename, 'a', compression='gzip', compression_opts=9, fillvalue=0.0)
            self.patches = self.hfile['patches']
            self.positions = self.hfile['positions']
            self.image_index = self.hfile['image_index']
            self.keys = self.hfile['keys']
            self.key_set = set(self.keys)
        else:
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
            self.patches.attrs['n_patches'] = patches

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

def extract_decaf(input_dir, output_dir, network_data_dir, files, num_patches, patch_size, image_dim, levels, oversample, layer_name, decaf_oversample, extraction_method):
    log = get_logger()

    #ex = DecafExtractor.DecafExtractor(layer_name)
    ex = CaffeExtractor.CaffeExtractor(layer_name, 
				       network_data_dir + 'hybridCNN_iter_700000_upgraded.caffemodel', 
				       network_data_dir + 'hybridCNN_deploy_no_relu_upgraded.prototxt', 
				       network_data_dir + 'hybrid_mean.npy')
    ex.set_parameters(patch_size, num_patches, levels, image_dim, decaf_oversample, extraction_method)

    if oversample:
        log.info('Extracting with zooming & rotations!')
        xforms_A = [DecafExtractor.Zoom(1), DecafExtractor.Zoom(2)]
        xforms_B = [DecafExtractor.Rotation(a) for a in range(-120,0,30) + range(30,150,30)]
        xforms = map(lambda (x,y): DecafExtractor.CombinedTransform(x,y), product(xforms_A, xforms_B))
        ex.transforms.extend(xforms)


    ds = Dataset(input_dir, output_dir, len(files),
                 num_patches * ex.get_number_of_features_per_image(),
                 'DECAF', patch_dim=ex.get_descriptor_size(),
                 patch_type='float', pos_type='uint16')

    for f in files:
        if f in ds:
            log.info('Skipping <%s>. Already in the dataset.', basename(f))
            continue

        features = ex.extract_image(f)

        if features.cursor > 0:
            (patches, positions) = features.get()

            ds.append(f, patches, positions)


if __name__ == '__main__':
    init_logging()
    log = get_logger()

    args = get_arguments()

    # Determining image files to extract from
    files = [ f for f in glob( join(args.input_dir, '*') )
                if splitext(f.lower())[1] in ['.jpg', '.jpeg'] ]

    # Determining which function to use for extraction
    if args.descriptor == 'DECAF':
        extract = extract_decaf
    else:
        raise Error, 'Only DECAF descriptor is supported.'
    if args.split >= 0:
        train_ix = (0, args.num_train_images)
        test_ix = (args.num_train_images, args.num_train_images + args.num_test_images)

        # Compiling directory name that will hold extracted descriptors
        tag = get_desc_name(dict(descriptor = args.descriptor,
                                 patches_per_image = args.patches,
                                 patch_size = args.patch_size,
                                 levels = args.levels,
                                 image_dim = args.image_dim,
                                 num_train_images = args.num_train_images,
                                 num_test_images = args.num_test_images,
                                 oversample=args.oversample,
                                 decaf_layer=args.layer_name,
                                 decaf_oversample=args.decaf_oversample
                                 ))



        # Extracting split
        random.shuffle(files)

        train_files = files[train_ix[0]:train_ix[1]]
        test_files = files[test_ix[0]:test_ix[1]]

        train_files = train_files[:args.num_train_images]
        test_files = test_files[:args.num_test_images]

        # Checking for train/test file overlap
        assert(len(set(train_files).intersection(set(test_files))) == 0)

        log.info('Extracting from training files...')
        output_dirname = join(args.output_dir, 'train', 'split_%d' % args.split)
        extract(args.input_dir, output_dirname, args.network_data_dir, train_files, args.patches, args.patch_size, args.image_dim, args.levels, args.oversample, args.layer_name, args.decaf_oversample, args.patch_method)

        log.info('Extracting from testing files...')
        output_dirname = join(args.output_dir, 'test', 'split_%d' % args.split)
        extract(args.input_dir, output_dirname, args.network_data_dir, test_files, args.patches, args.patch_size, args.image_dim, args.levels, args.oversample, args.layer_name, args.decaf_oversample, args.patch_method)
    else:
        log.info('Extracting all files...')
        output_dirname = args.output_dir
        extract(args.input_dir, output_dirname, args.network_data_dir, files, args.patches, args.patch_size, args.image_dim, args.levels, args.oversample, args.layer_name, args.decaf_oversample, args.patch_method)


