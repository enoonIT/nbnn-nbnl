from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename, exists
from common import init_logging, get_logger

import numpy as np
from h5py import File as HDF5File

def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='Descriptor checker utility.')
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Directory with HDF5 files.")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    init_logging()
    log = get_logger()

    args = get_arguments()

        # Determining image files to extract from
    files = [ f for f in glob( join(args.input_dir, '*') )
                if splitext(f.lower())[1] in ['.hdf5'] ]
    global_average_patches = 0
    for f in files:
        average_patches = 0
        print "Checking " + f
        dataset = HDF5File(f,"r")
        keys = dataset["keys"]
        image_index = dataset["image_index"]
        if(keys.len()!=image_index.len()):
            print "Dataset mismatch"
        x = 0
        print "Contains " + str(len(keys)) + " images"
        for k in keys:
            if(not k.strip()):
                print "Empty key: " + str(x)
            x += 1

        x = 0
        for imInd in image_index:
            nPatches = imInd[1]-imInd[0]
            if(nPatches < 10):
                print "Low patch count for :" + str(x)
            x += 1
            average_patches += nPatches
        average_patches = average_patches/x
        global_average_patches += average_patches
        print "Average patches: " + str(average_patches)
        patches = dataset["patches"][:]
        if(np.all(patches>=0) ):
            print "RELU!"
        else:
            print "Not RELU"

        dataset.close()
    global_average_patches = global_average_patches/len(files)
    print "Global average of patches: " + str(global_average_patches)