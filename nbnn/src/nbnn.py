from h5py import File as hfile
import glob
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import PCA
import pyflann
from sklearn.neighbors import LSHForest

class options(object):
    use_position = None
    pca = None


def get_arguments():
    parser = ArgumentParser(description='Utility to execute NBNN with or without PCA.')
    parser.add_argument("train_folder", help="The folder containing the training features")
    parser.add_argument("test_folder", help="The folder containing the testing features")
    parser.add_argument("--pca", help="The number of components to keep in PCA", type=int)
    parser.add_argument("-p", "--position", help="Wheter to append patch position to the features", action="store_true")
    args = parser.parse_args()
    if args.pca:
        options.pca = args.pca
    if args.position:
        options.use_position = True
    return args

class NN_Engine: #faster but less accurate
    def __init__(self, num_neighbors=1):
        self.engine = pyflann.FLANN()
        self.num_neighbors = num_neighbors
    def fit(self, X):
        self.engine.build_index(X, algorithm='kdtree', trees=4)
    def dist(self, X):
        (_, rs) = self.engine.nn_index(X, num_neighbors=self.num_neighbors)
        return rs

class LHSForestEngine:

    def __init__(self):
        self.engine = LSHForest(random_state=42)
    def fit(self, data):
        self.engine.fit(data)
    def dit(self, data):
        distances, indices = lshf.kneighbors(data, n_neighbors=1)
        return distances

def load_patches(folder):
    #import pdb; pdb.set_trace()
    files = glob.glob(folder + "*.hdf5")
    num_classes = len(files)
    print("Loading " + str(num_classes) + " classes from " + folder)
    all_features = []
    for pfile in files:
        f = hfile(pfile)
        iid = f["image_index"]
        class_patches = f["patches"][0:iid[:].max(), :]
        if options.use_position:
            class_positions = f["positions"][0:iid[:].max(), :]
            class_patches = np.hstack([class_patches, class_positions])
        all_features.append(class_patches)
    return all_features

def do_nbnn(train_folder, test_folder):
    train = load_patches(args.train_folder)
    test = load_patches(args.test_folder)
    if options.pca:
        pca = PCA(n_components=options.pca)
        pca.fit(np.vstack(train))
        print("Keeping " + str(pca.explained_variance_ratio_.sum()) + " variance (" + str(options.pca) + ") components")
    for (idx, patches) in enumerate(train):


if __name__ == '__main__':
    args = get_arguments()