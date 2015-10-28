#!/usr/bin/python

########################################
# Patch/descriptor extraction utility. #
#                                      #
# Author: ilja.kuzborskij@idiap.ch     #
########################################

from argparse import ArgumentParser
from os import makedirs
from os.path import join
from common import init_logging, get_logger
import os

from h5py import File as HDF5File

import CaffeExtractorPlus
from collections import namedtuple


def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='Patch/descriptor extraction utility.')
    parser.add_argument("--patches", dest="patches", type=int, default=100,
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

    return args

def create_hdf5_dataset(output_filename, patches, positions):
    log = get_logger()
    log.debug('Saving extracted descriptors to %s', output_filename)
    hfile = HDF5File(output_filename, 'w', compression='gzip', fillvalue=0.0)
    hpatches = hfile.create_dataset('patches', patches.shape, dtype="float32", chunks=True)
    hpositions = hfile.create_dataset('positions', positions.shape, dtype="uint16", chunks=True)
    hpatches[:]=patches
    hpositions[:]=positions
    hfile.close()

def touch(fname):
    get_logger().info("Creating " + fname)
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

def walk(params, dir_name, files):
    input_dir = params.input_dir
    out_dir = params.output_dir
    ex = params.extractor
    common_prefix = os.path.commonprefix([dir_name, input_dir])
    relpath = os.path.relpath(dir_name, common_prefix)
    new_rel_folder = join(out_dir, relpath)
    if not os.path.exists(new_rel_folder):
        get_logger().info("Creating folder " + new_rel_folder)
        makedirs(new_rel_folder)
    get_logger().info("Contains " + str(files))
    for f in files:
        get_logger().info("Parsing " + join(dir_name,f))
        new_file = join(new_rel_folder, f + ".hdf5")
        old_file = join(dir_name,f)
        if os.path.isdir(old_file):
            continue
        elif not is_image(old_file):
            get_logger().info("Skipping " + old_file + ": not an image")
            continue
        elif os.path.exists(new_file):
            get_logger().info("Skipping " + new_file + ": already exists")
            continue
        features = ex.extract_image(old_file)
        (patches6, patches7, positions) = features.get()
        create_hdf5_dataset(new_file, patches7, positions)




def extract(input_dir, output_dir, network_data_dir, num_patches, patch_size, image_dim, levels, layer_name):
    log = get_logger()
    BATCH_SIZE = 16
    log.info("Walking " + input_dir)
    ex = CaffeExtractorPlus.CaffeExtractorPlus(
                       network_data_dir + 'hybridCNN_iter_700000_upgraded.caffemodel',
                       network_data_dir + 'hybridCNN_deploy_no_relu_upgraded.prototxt',
                       network_data_dir + 'hybrid_mean.npy')
    ex.set_parameters(patch_size, num_patches, levels, image_dim, BATCH_SIZE)
    params = namedtuple("Params","input_dir output_dir extractor")
    os.path.walk(input_dir, walk, params(input_dir, output_dir, ex))


if __name__ == '__main__':
    init_logging()
    log = get_logger()

    args = get_arguments()
    log.info('Extracting all files...')
    output_dirname = args.output_dir
    extract(args.input_dir, args.output_dir, args.network_data_dir, args.patches, args.patch_size, args.image_dim, args.levels,args.layer_name)


