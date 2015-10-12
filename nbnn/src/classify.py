#!/usr/bin/python

from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename, exists, dirname
import os
from common import get_logger, init_logging
from subprocess import call
import random

import numpy as np
from h5py import File as HDF5File
import pyflann
# from sklearn.neighbors import KernelDensity as KDE
# from sklearn import preprocessing
# from sklearn import svm, linear_model
# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.utils import compute_class_weight
# from sklearn.cross_validation import StratifiedKFold

PATCH_TYPE = 'patches'

def get_arguments():
    log = get_logger()

    parser = ArgumentParser(description='NN support selection and classification tool.')
    parser.add_argument("--train-dir", dest="train_dir",
                        help="Directory containing training HDF5 files.")
    parser.add_argument("--test-dir", dest="test_dir",
                        help="Directory containing testing HDF5 files.")
    parser.add_argument("--support", dest="support",
                        help="Directory or file to store/get NN support.")
    parser.add_argument("--result-dir", dest="result_dir",
                        help="Directory to store NN distances.")
    parser.add_argument("--support-size", dest="support_size", type=int,
                        help="Support size to select from each class.")
    parser.add_argument("--num-train-images", dest="num_train_images", type=int,
                        help="Number of images to use from training set.")
    parser.add_argument("--num-test-images", dest="num_test_images", type=int,
                        help="Number of images to use from the test set.")
    parser.add_argument("--gamma", dest="gamma", type=float,
                        help="KDE bandwidth.")
    parser.add_argument("--knn", dest="knn", type=int, default=1,
                        help="Number of nearest neighbors to look for.")
    parser.add_argument("--alpha", dest="alpha", type=float, default=0,
                        help="Patch position influence.")
    parser.add_argument("--cmd", dest="cmd",
                        choices=['select-random', 'classify'],
                        help="Command to execute.")
    parser.add_argument("--alg-type", dest="alg_type", default='nn',
                        choices=['nn', 'kde'],
                        help="Nearest neighbor algorithm type.")
    parser.add_argument("--on_the_fly_splits", dest="on_the_fly_splits",
                        action='store_true',
                        help="Splits are computed on the fly.")
    parser.add_argument("--overwrite", dest="overwrite",
                        action='store_true',
                        help="Overwrite result of command (if any).")
    parser.add_argument("--patch_name", dest="patch_name",
                        help="The name of the patches in the HDF5 File.")

    args = parser.parse_args()

    if not 'cmd' in args:
        log.error('cmd option is required, but not present.')
        exit()

    if args.cmd is 'train' and not 'train_dir' in args:
        log.error('train-dir option is required, but not present.')
        exit()

    if not 'support' in args:
        log.error('support option is required, but not present.')
        exit()

    if args.cmd is 'nn' and not 'result_dir' in args:
        log.error('resilt-dir option is required, but not present.')
        exit()

    return args


def select_random_support(train_dir, support_dir, num_train_images, support_size, position_influence):
    log = get_logger()

    train_files = [ f for f in glob( join(train_dir, '*') )
                    if splitext(f.lower())[1] == '.hdf5' ]

    try:
        os.makedirs(support_dir)
    except:
        pass

    for target_file in train_files:
        log.info('Extracting random support from "%s"...', basename(target_file))
        #(patches, _)= get_standardized_patches(target_file, num_train_images, position_influence)
        (patches, _)= get_patches(target_file, num_train_images, position_influence)
        rand_ix = random.sample(range(patches.shape[0]), min(patches.shape[0], support_size))
        patches = patches[np.array(rand_ix), :]

        fh = HDF5File(join(support_dir, basename(target_file)), 'w')
        ds = fh.create_dataset('support', patches.shape, dtype='float')
        ds[:] = patches
        ds.attrs['cursor'] = patches.shape[0]

        fh.close()

def get_num_patches(filename, num_images):
    hfile = HDF5File(filename, 'r')
    total_num_patches = hfile[PATCH_TYPE].attrs['cursor']
    dim = hfile[PATCH_TYPE].shape[1]

    if num_images == 0:
        num_images = hfile['image_index'].shape[0]

    image_index = hfile['image_index'][:min(num_images, hfile['image_index'].shape[0])]

    # Getting patches only from desired number of images
    num_patches = min(image_index[-1,1], total_num_patches)

    hfile.close()
    return (num_patches, dim)

def get_patches(filename, num_images, position_influence=0):
    hfile = HDF5File(filename, 'r')
    total_num_patches = hfile[PATCH_TYPE].attrs['cursor']

    if num_images == 0:
        num_images = hfile['image_index'].shape[0]

    image_index = hfile['image_index'][:min(num_images, hfile['image_index'].shape[0])]

    # Getting patches only from desired number of images
    num_patches = min(image_index[-1,1], total_num_patches)
    patches = hfile[PATCH_TYPE][:num_patches, :]

    # patches = patches.astype(float)
    # norms = (patches**2).sum(axis=1)**0.5
    # patches /= norms.max()

    #patches = patches.astype(float)
    #patches /= patches.max()

    feature_type = hfile[PATCH_TYPE].attrs.get('feature_type', None)

    # if feature_type == 'DECAF':
    #     norms = (patches**2).sum(axis=1)**0.5
    #     patches /= norms.max()

    if position_influence > 0:
        pos = hfile['positions'][:num_patches, :]

        pos = pos.astype(float)
        max_x = pos[:, 0].max()
        max_y = pos[:, 1].max()

        if max_x > 0:
            pos[:, 0] /= max_x
        if max_y > 0:
            pos[:, 1] /= max_y

        patches = np.hstack([patches, position_influence * pos])

    hfile.close()
    return (patches, image_index)

def get_support(filename, size):
    hfile = HDF5File(filename, 'r')
    ds = hfile['support']
    patches = ds[:min(size, ds.shape[0]), :]

    hfile.close()
    return patches.astype('float')

def is_selected_support(filename):
    hfile = HDF5File(filename, 'r')
    return 'support' in hfile.keys()

class KDE_Engine:
    def __init__(self, gamma, **args):
        self.gamma = gamma
    def fit(self, X):
        self.engine = KDE(bandwidth= (1.0/(2*self.gamma))**0.5 )
        self.engine.fit(X)
    def dist(self, X):
        rs = -self.engine.score_samples(X)
        return rs

class NN_Engine:
    def __init__(self, num_neighbors=1):
        self.engine = pyflann.FLANN()
        self.num_neighbors = num_neighbors
    def fit(self, X):
        self.engine.build_index(X, algorithm='kdtree', trees=4)
    def dist(self, X):
        (_, rs) = self.engine.nn_index(X, num_neighbors=self.num_neighbors)
        return rs

def get_engine(alg_type, **params):
    if alg_type == 'nn':
        return NN_Engine(num_neighbors=params.get('knn', 1))
    elif alg_type == 'kde':
        return KDE_Engine(**params)

def loadSplits(patch_folder, nTrain, nTest, position_influence):
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







def on_the_fly_classify(engine, test_dir, support_dir, num_train_images, num_test_images, position_influence, support_size=0):
    log = get_logger()

    test_files = sorted(glob( join(test_dir, '*.hdf5') ), key=basename)
    num_classes = len(test_files)
    log.info('Testing w.r.t. %d classes.' % num_classes)
    if position_influence > 0:
        log.info('Position influence (alpha) is %.2f.', position_influence)
    # Allocating distances for each test class
    dists = np.ndarray( (num_classes, num_classes, num_test_images) )

    # Identifying labels
    labels = np.vstack([c*np.ones((1,num_test_images), dtype=np.int) for c in range(num_classes)])

    log.info('Looking for nearest neighbors...')
    for (support_class,f) in enumerate(test_files):
        support_filename = join(support_dir, basename(f))

        if is_selected_support(support_filename):
            support = get_support(support_filename, support_size)
        else:
            support, _ = get_patches(support_filename, num_train_images, position_influence)

        # Creating index for current class
        log.info('\tBuilding index from support of class "%s"...', basename(f))
        engine.fit(support)
        del support

        # Evaluating test samples for all classes using current index
        for (test_class, test_filename) in enumerate(test_files):
            (test_patches, test_image_index) = get_patches(test_filename, num_test_images, position_influence)

            log.info('\tLooking for NNs of "%s"...', basename(test_filename))
            im_to_class_dists = engine.dist(test_patches)

            if len(im_to_class_dists.shape) > 1: # In case of k-NN, we average
                im_to_class_dists = im_to_class_dists.mean(axis=1)

            dists[support_class, test_class, :] = \
                np.array([sum(im_to_class_dists[ix[0]:ix[1]]) for ix in test_image_index])


    predictions = dists.argmin(axis=0)
    acc = (labels == predictions).mean()
    log.info('*** Recognition accuracy is: %.2f%%', acc*100)

    return acc

def classify_with_support(engine, test_dir, support_dir, num_train_images, num_test_images, position_influence, support_size=0):
    log = get_logger()

    test_files = sorted(glob( join(test_dir, '*.hdf5') ), key=basename)
    num_classes = len(test_files)

    log.info('Testing w.r.t. %d classes.' % num_classes)
    if position_influence > 0:
        log.info('Position influence (alpha) is %.2f.', position_influence)

    # Allocating distances for each test class
    dists = np.ndarray( (num_classes, num_classes, num_test_images) )

    # Identifying labels
    labels = np.vstack([c*np.ones((1,num_test_images), dtype=np.int) for c in range(num_classes)])

    log.info('Looking for nearest neighbors...')
    for (support_class,f) in enumerate(test_files):
        support_filename = join(support_dir, basename(f))

        if is_selected_support(support_filename):
            support = get_support(support_filename, support_size)
        else:
            support, _ = get_patches(support_filename, num_train_images, position_influence)

        # Creating index for current class
        log.info('\tBuilding index from support of class "%s"...', basename(f))
        engine.fit(support)
        del support

        # Evaluating test samples for all classes using current index
        for (test_class, test_filename) in enumerate(test_files):
            (test_patches, test_image_index) = get_patches(test_filename, num_test_images, position_influence)

            log.info('\tLooking for NNs of "%s"...', basename(test_filename))
            im_to_class_dists = engine.dist(test_patches)

            if len(im_to_class_dists.shape) > 1: # In case of k-NN, we average
                im_to_class_dists = im_to_class_dists.mean(axis=1)

            dists[support_class, test_class, :] = \
                np.array([sum(im_to_class_dists[ix[0]:ix[1]]) for ix in test_image_index])


    predictions = dists.argmin(axis=0)
    acc = (labels == predictions).mean()
    log.info('*** Recognition accuracy is: %.2f%%', acc*100)

    return acc

if __name__ == '__main__':
    init_logging()

    args = get_arguments()
    PATCH_TYPE = args.patch_name
    if args.cmd == 'select-random':
        select_random_support(args.train_dir, args.support, args.num_train_images,
                              args.support_size, args.alpha)
    elif args.cmd == 'classify':
        classify = classify_with_support
        if(args.on_the_fly_splits):
            classify = on_the_fly_classify
        classify(get_engine(args.alg_type, gamma=args.gamma, knn=args.knn),
                              args.test_dir, args.support,
                              args.num_train_images, args.num_test_images,
                              args.alpha, args.support_size)

