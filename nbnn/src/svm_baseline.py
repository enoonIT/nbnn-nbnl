# -*- coding: utf-8 -*-
from sklearn import svm
import numpy as np
from h5py import File as HDF5File
from argparse import ArgumentParser
from glob import glob
from os.path import join, basename
from common import init_logging, get_logger
import time
from collections import namedtuple

PatchOptions = namedtuple("PatchOptions","patch_name size")
patchOptions = PatchOptions("patches7", 4096)

def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='SVM based classification for whole images.')
    parser.add_argument("--input-dir", dest="input_dir",
                        help="Directory containing HDF5 files.")
    parser.add_argument("--num-train-images", dest="num_train_images", type=int,
                        help="Number of images to use from training set.")
    parser.add_argument("--num-test-images", dest="num_test_images", type=int,
                        help="Number of images to use from the test set.")
    parser.add_argument("--patch_name", dest="patch_name",
                        help="The name of the patches in the HDF5 File.")
    args = parser.parse_args()
    if not 'input_dir' in args:
        log.error('input dir is required, but not present.')
        exit()
    if not 'num_train_images' in args:
        log.error('num_train_images is required, but not present.')
        exit()
    if not 'num_test_images' in args:
        log.error('num_test_images is required, but not present.')
        exit()
    return args


def load_split(input_folder, nTest, nTrain):
    logger = get_logger()
    files = sorted(glob( join(input_folder, '*.hdf5') ), key=basename)
    nClasses = len(files);
    logger.info("Loading " + str(nClasses) + " classes")
    train_patches = np.empty([nClasses*nTrain, patchOptions.size]) # nClasses*nSamples x nFeatures
    test_patches = np.empty([nClasses*nTest, patchOptions.size])
    train_labels = np.empty([nClasses*nTrain])
    test_labels = np.empty([nClasses*nTest])
    start = time.clock()
    train_patch_count = test_patch_count = 0
    for (classNumber,filename) in enumerate(files):
        hfile = HDF5File(filename, 'r')
        iid = hfile["image_index"][:]
        nImages = iid.shape[0]
        assert nImages >= (nTrain + nTest), "Not enough images!"
        np.random.shuffle(iid)
        trainIdx = iid[0:nTrain]
        testIdx  = iid[nTrain:nTrain+nTest]
        patches = hfile[patchOptions.patch_name][:]
        for iid in trainIdx:
            train_patches[train_patch_count]=patches[iid[0]]
            train_patch_count += 1
        train_labels[classNumber*nTrain:(classNumber+1)*nTrain]=classNumber*np.ones(nTrain)
        for iid in testIdx:
            test_patches[test_patch_count]=patches[iid[0]]
            test_patch_count += 1
        test_labels[classNumber*nTest:(classNumber+1)*nTest]=classNumber*np.ones(nTest)
        logger.info("Patch count: " + str(train_patch_count) + " training and " + str(test_patch_count) + " test patches for class " + filename)
    end = time.clock()
    logger.info("It took " + str((end-start)) + " seconds");
    LoadedData = namedtuple("LoadedData","train_patches train_labels test_patches test_labels")
    return LoadedData(train_patches, train_labels, test_patches, test_labels)


if __name__ == '__main__':
    init_logging()
    logger = get_logger()
    args = get_arguments()
    loaded_data = load_split(args.input_dir, args.num_test_images, args.num_train_images)
    kernels = ['linear','poly','rbf','sigmoid']
    for k in kernels:
        logger.info("Fitting SVM to data with " + k + " kernel")
        clf = svm.SVC(kernel=k)
        start=time.clock()
        clf.fit(loaded_data.train_patches, loaded_data.train_labels)
        res = clf.predict(loaded_data.test_patches)
        correct = (res==loaded_data.test_labels).sum()
        end=time.clock()
        logger.info("Got " + str((100.0*correct)/loaded_data.test_labels.size) + "% correct, took " + str(end-start) + " seconds")