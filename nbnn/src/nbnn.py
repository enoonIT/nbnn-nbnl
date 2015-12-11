from h5py import File as hfile
import glob
from argparse import ArgumentParser
import numpy as np
from sklearn.decomposition import RandomizedPCA
import pyflann
from sklearn.neighbors import LSHForest, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from common import get_logger, init_logging


class options(object):
    use_position = None
    pca = None
    scale = None
    relu = None


class ImageClass:

    def __init__(self, patches, name, iid):
        self.patches = patches
        self.iid = iid
        self.name = name


def get_arguments():
    parser = ArgumentParser(description='Utility to execute NBNN with or without PCA.')
    parser.add_argument("train_folder", help="The folder containing the training features")
    parser.add_argument("test_folder", help="The folder containing the testing features")
    parser.add_argument("--pca", help="The number of components to keep in PCA", type=int)
    parser.add_argument("-p", "--position", help="Wheter to append patch position to the features", action="store_true")
    parser.add_argument("-sc", "--scale", help="Apply normalization to the data", action="store_true")
    parser.add_argument("--relu", help="Apply ReLu to features", action="store_true")
    args = parser.parse_args()

    options.pca = args.pca
    options.use_position = args.position
    options.scale = args.scale
    options.relu = args.relu
    return args


class NN_Engine:  # faster but less accurate

    def __init__(self, num_neighbors=1):
        self.engine = pyflann.FLANN()
        self.num_neighbors = num_neighbors
        self.name = "FLANN"
    def fit(self, X):
        self.engine.build_index(X, algorithm='kdtree', trees=4)

    def dist(self, X):
        (_, rs) = self.engine.nn_index(X, num_neighbors=self.num_neighbors)
        return rs


class LHSForestEngine:

    def __init__(self):
        self.engine = LSHForest(random_state=42)
        self.name = "LHS"

    def fit(self, data):
        self.engine.fit(data)

    def dist(self, data):
        distances, indices = self.engine.kneighbors(data, n_neighbors=1)
        return distances.ravel()


class KdeEngine:

    def __init__(self):
        self.engine = NearestNeighbors(algorithm='kd_tree', n_neighbors=1)
        self.name = "KDE"

    def fit(self, data):
        self.engine.fit(data)

    def dist(self, data):
        distances, indices = self.engine.kneighbors(data)
        return distances.ravel()


def load_patches(folder):
    #import pdb; pdb.set_trace()
    files = glob.glob(folder + "*.hdf5")
    num_classes = len(files)
    get_logger().info("Loading " + str(num_classes) + " classes from " + folder)
    all_features = []
    for pfile in files:
        f = hfile(pfile)
        iid = f["image_index"]
        class_patches = f["patches"][0:iid[:].max(), :]
        if options.use_position:
            class_positions = f["positions"][0:iid[:].max(), :]
            class_patches = np.hstack([class_patches, class_positions])
        all_features.append(ImageClass(class_patches, pfile, iid[:]))
    return all_features


def do_nbnn(train_folder, test_folder):
    train = load_patches(args.train_folder)
    test = load_patches(args.test_folder)
    if options.relu:
        get_logger().info("Applying RELU")
        for class_data in train:
            class_data.patches = class_data.patches.clip(min=0)
        for class_data in test:
            class_data.patches = class_data.patches.clip(min=0)
    if options.scale:
        get_logger().info("Applying standardization")
        scaler = StandardScaler(copy=False)
        scaler.fit(np.vstack([t.patches for t in train]))
        for class_data in train:
            class_data.patches = scaler.transform(class_data.patches)
        for class_data in test:
            class_data.patches = scaler.transform(class_data.patches)
    if options.pca:
        get_logger().info("Calculating PCA")
        pca = RandomizedPCA(n_components=options.pca)
        pca.fit(np.vstack([t.patches for t in train]))
        #for class_data in train:
            #get_logger().info("Fitting class " + class_data.name)
            #pca.partial_fit(class_data.patches)
        get_logger().info("Keeping " + str(pca.explained_variance_ratio_.sum()) + " variance (" + str(options.pca) +
             ") components\nApplying PCA")
        for class_data in train:
            class_data.patches = pca.transform(class_data.patches)
        for class_data in test:
            class_data.patches = pca.transform(class_data.patches)
    nbnn(train, test, NN_Engine())
    #nbnn(train, test, LHSForestEngine())
    #nbnn(train, test, KdeEngine())


def nbnn(train, test, engine):
    num_classes = len(train)
    num_test_images = len(test[0].iid)
    dists = np.ndarray((num_classes, num_classes, num_test_images))
    # Identifying labels
    labels = np.vstack([c * np.ones((1, num_test_images), dtype=np.int) for c in range(num_classes)])
    for (support_class, class_data) in enumerate(train):
        get_logger().info("Loading class " + class_data.name + " as support - " + str(class_data.patches.shape))
        engine.fit(class_data.patches)
        for test_class, test_data in enumerate(test):
            #get_logger().info("Testing class " + test_data.name)
            im_to_class_dists = engine.dist(test_data.patches)
            #import pdb; pdb.set_trace()
            dists[support_class, test_class, :] = \
                np.array([sum(im_to_class_dists[ix[0]:ix[1]]) for ix in test_data.iid])
    predictions = dists.argmin(axis=0)
    acc = (labels == predictions).mean()
    get_logger().info('*** Recognition accuracy by ' + engine.name + ' is: ' + str(acc * 100))


if __name__ == '__main__':
    args = get_arguments()
    init_logging()
    do_nbnn(args.train_folder, args.test_folder)