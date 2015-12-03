from argparse import ArgumentParser
from os.path import join
import os, re, common
from sklearn import svm
from h5py import File as HDF5File
import time
import numpy as np
nregex = re.compile("(\w+)\_\d")

class fakeLogger():
    def info(self, stuff):
        print stuff

def init_logging():
    common.init_logging()
    print "Started logging"

def get_logger():
    #logger = fakeLogger()
    logger = common.get_logger()
    return logger

def get_arguments():
    parser = ArgumentParser(description='Utility to load training\testing splits for the Washington RGB-D Object Dataset.')
    parser.add_argument("input_folder",help="The folder containing the extracted features")
    parser.add_argument("split_file",help="The file from which to load the splits")
    parser.add_argument("split",help="Which training split to load",type=int)
    args = parser.parse_args()
    return args

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def get_testing_folders(split_file, split):
    """Returns a list containing the object instances we want to keep for testing

    >>> get_testing_folders('nofile',1)
    Traceback (most recent call last):
        ...
    IOError: [Errno 2] No such file or directory: 'nofile'
    """
    log = get_logger()
    instances = []
    with open(split_file) as sfile:
        data = sfile.read()
        trial = re.split("\*+\strial\s\d+\s\*+\n",data)[split]
        instances = filter(None, trial.splitlines())
        log.info("Loaded split test instances list")
        log.info(instances)
        log.info("There are " + str(len(instances)) +" test instances")
    return instances

def is_hdf5(filename):
    return filename.lower().endswith('.hdf5')

def is_depth_file(filename):
    return filename.lower().endswith('_depthcrop.png.hdf5')

train_features = {}
test_features = {}

def walk(test_instances, dir_name, files):
    folder_name = os.path.basename(os.path.normpath(dir_name))
    if not hasNumbers(folder_name): #we are only interested in instance level folders (they contain numbers)
        return
    basename =  nregex.match(folder_name).group(1)

    features = train_features
    if folder_name in test_instances:
        features = test_features;

    get_logger().info(folder_name + " " + basename)
    image_features = features.get(basename, [])
    #import pdb; pdb.set_trace()
    for f in files:
        file_path = join(dir_name,f)
        if os.path.isdir(file_path):
            continue
        elif not is_hdf5(file_path):
            get_logger().info("Skipping " + file_path + ": not an Hdf5 file")
            continue
        elif not is_depth_file(file_path):
            #get_logger().info("Skipping " + file_path + ": not a depth file")
            continue
        with HDF5File(file_path,"r") as hfile:
            image_features.append(hfile["patches"][:])
    features[basename]=image_features


if __name__ == '__main__':
    init_logging()
    log = get_logger()
    args = get_arguments()
    test_instances = get_testing_folders(args.split_file, args.split)
    os.path.walk(args.input_folder, walk, test_instances)

    log.info("Transforming train data into numpy format")
    tr_patches = np.vstack([np.vstack(train_features[c]) for c in sorted(train_features)])
    c_n = 0
    trl = []
    for c in sorted(train_features):
        trl.append(c_n * np.ones(len(train_features[c])))
        c_n+=1
    te_patches = np.vstack([np.vstack(test_features[c]) for c in sorted(test_features)])
    c_n = 0
    tel = []
    for c in sorted(test_features):
        tel.append(c_n * np.ones(len(test_features[c])))
        c_n+=1
    #import pdb; pdb.set_trace()
    tr_labels = np.concatenate(trl)
    te_labels = np.concatenate(tel)
    clf = svm.LinearSVC(dual=False)
    start=time.clock()
    log.info("Fitting SVM")
    clf.fit(tr_patches, tr_labels)
    log.info("Predicting labels")
    acc = clf.score(te_patches, te_labels)
    log.info("Accuracy: " + str(acc))