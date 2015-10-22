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
from ClassPatches import ClassPatches
from joblib import Parallel, delayed

class patchOptions(object):
    patch_name="patches7"
    size=4096

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
    parser.add_argument("--patches-per-image", dest="patches_per_image", type=int,
                        help="Number of patches for each image.")
    parser.add_argument("--cmd", dest="cmd",
                        choices=['whole-image-svm', 'svm-nbnl'],
                        help="Command to execute.")
    args = parser.parse_args()
    patchOptions.patch_name=args.patch_name
    if not 'input_dir' in args:
        log.error('input dir is required, but not present.')
        exit()
    if not 'cmd' in args:
        log.error('cmd is required, but not present.')
        exit()
    if not 'num_train_images' in args:
        log.error('num_train_images is required, but not present.')
        exit()
    if not 'num_test_images' in args:
        log.error('num_test_images is required, but not present.')
        exit()
    return args


def load_split_whole_image_only(input_folder, nTrain, nTest):
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
        hfile.close()
    end = time.clock()
    logger.info("It took " + str((end-start)) + " seconds");
    LoadedData = namedtuple("LoadedData","train_patches train_labels test_patches test_labels")
    return LoadedData(train_patches, train_labels, test_patches, test_labels)

def get_indexes(patch_folder, nTrain, nTest, position_influence):
    files = sorted(glob( join(patch_folder, '*.hdf5') ), key=basename)
    train = []
    test = []
    for (classNumber,filename) in enumerate(files):
        #support_filename = join(".", basename(filename))
        hfile = HDF5File(filename, 'r')
        iid = hfile["image_index"][:]
        nImages = iid.shape[0]
        assert nImages >= (nTrain + nTest), "Not enough images!"
        np.random.shuffle(iid)
        trainIdx = iid[0:nTrain]
        testIdx  = iid[nTrain:nTrain+nTest]
        trainData = ClassPatches(filename, trainIdx, patchOptions.patch_name)
        train.append(trainData) #train data is actually loaded only when needed
        testData = ClassPatches(filename, testIdx, patchOptions.patch_name)
        test.append(testData)
        hfile.close()
    Data = namedtuple("Data","Train Test")
    return Data(train, test)

def do_nbnl(args):
    logger = get_logger()
    logger.info("Getting indexes")
    data = get_indexes(args.input_dir, args.num_train_images, args.num_test_images, args.patches_per_image)
    train = data.Train
    num_classes = len(train)
    logger.info("Loading training patches")
    X = np.vstack([t.get_patches() for t in train])
    for t in train: t.unload()
    Y = np.vstack([c*np.ones((train[c].get_num_patches(),1), dtype=np.int) for c in range(num_classes)])
    clf = svm.LinearSVC(dual=False)
    logger.info("Training Linear SVM at patch level")
    logger.info(str(X.shape) + " X, " + str(Y.shape) + " Y")
    clf.fit(X,Y.ravel())
    logger.info("Training completed, freeing training patches")
    del X, Y
    test = data.Test
    testX = np.vstack([t.get_patches() for t in test])
    for t in test: t.unload()
    testY = np.vstack([c*np.ones((test[c].get_num_patches(),1), dtype=np.int) for c in range(num_classes)])
    logger.info(str(testX.shape) + " testX, " + str(testY.shape) + " testY")
    logger.info("Evaluating test patches...")
    confidence = clf.decision_function(testX)
    predicted = np.argmax(confidence,1)
    correct=(predicted==testY).sum()
    score = clf.score(testX, testY)
    logger.info("Accuracy " + str(score) + " at patch level " + str(correct/len(predicted)))


def do_whole_image_svm(args):
    logger = get_logger()
    loaded_data = load_split_whole_image_only(args.input_dir, args.num_train_images, args.num_test_images)
    #kernels = ['linear']
    #cVals = [0.1, 1, 10, 100, 1000]
    k='linear'
    c=1
    logger.info("Fitting SVM to data with " + k + " kernel and " + str(c) + " C val")
    clf = svm.SVC(C=c, kernel=k)
    start=time.clock()
    clf.fit(loaded_data.train_patches, loaded_data.train_labels)
    res = clf.predict(loaded_data.test_patches)
    correct = (res==loaded_data.test_labels).sum()
    score = clf.score(loaded_data.test_patches, loaded_data.test_labels)
    end=time.clock()
    logger.info("Got " + str((100.0*correct)/loaded_data.test_labels.size) + "% correct, took " + str(end-start) + " seconds " + str(score))

if __name__ == '__main__':
    init_logging()
    args = get_arguments()
    if args.cmd == 'whole-image-svm':
        do_whole_image_svm(args)
    elif args.cmd == "svm-nbnl":
        do_nbnl(args)