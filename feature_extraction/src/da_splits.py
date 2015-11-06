from collections import namedtuple
from common import *
from h5py import File as HDF5File
import numpy as np
from argparse import ArgumentParser
from os.path import join, basename
from os import makedirs

class patchOptions(object):
    patch_name="patches"
    position_name = "positions"
    relu = True
    patch_dim=4096

MiscData = namedtuple("MiscData","patches positions idx")
ClassData = namedtuple("ClassData","patches positions tags n_samples indexes")
ClassIndexes = namedtuple("ClassIndexes", "filename index")
Data = namedtuple("Data","train test")

def get_arguments():
    parser = ArgumentParser(description='HD5 Splitter.')
    parser.add_argument("--source-dir", dest="source_dir",
                        help="Directory with HDF5 images.")
    parser.add_argument("--target-dir", dest="target_dir",
                        help="Directory with HDF5 images.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory to put HDF5 files to.")
    parser.add_argument("--train-images", dest="train_images", type=int,
                        help="Number of train images.")
    parser.add_argument("--transfer-imgs", dest="transfer_imgs", type=int,
                        help="Number of transfer images.")
    parser.add_argument("--split", dest="split", type=int,
                        help="Split Number")
    args = parser.parse_args()

    return args

def load_patches(class_data):
        hfile = HDF5File(class_data.filename, 'r')
        patches = hfile[patchOptions.patch_name][:]
        positions = hfile[patchOptions.position_name][:]
        feature_dim = patchOptions.patch_dim
        indexes = class_data.index
        num_patches=(indexes[:,1]-indexes[:,0]).sum()
        loaded_patches = np.empty([num_patches, feature_dim])
        loaded_positions = np.empty([num_patches, 2])
        loaded_idx = np.empty([class_data.index.shape[0], 2])
        tags = np.zeros([num_patches,1])
        patch_start = n_image = 0
        #import pdb; pdb.set_trace()
        for n, iid in enumerate(indexes):
            n_patches = iid[1]-iid[0]
            loaded_patches[patch_start:patch_start+n_patches,:] = patches[iid[0]:iid[1],:]
            loaded_positions[patch_start:patch_start+n_patches,:] = positions[iid[0]:iid[1],:]
            loaded_idx[n,:] = np.array([patch_start, patch_start+n_patches])
            tags[patch_start] = 1
            patch_start += n_patches
            n_image += 1
        hfile.close()
        return ClassData(loaded_patches, loaded_positions, tags, num_patches, loaded_idx)

def get_indexes_da(source_folder, target_folder, nTrain, transferImages):
    get_logger().info("Loading indexes")
    allowed = ['backpack.hdf5', 'headphones.hdf5', 'monitor.hdf5', 'bike.hdf5', 'keyboard.hdf5', 'mouse.hdf5', 'projector.hdf5', 'calculator.hdf5', 'laptop.hdf5', 'mug.hdf5']
    train = []
    test = []
    MAX_TEST = 100
    for categ in allowed:
        #support_filename = join(".", basename(filename))
        s_filename = join(source_folder, categ)
        t_filename = join(target_folder, categ)
        get_logger().info("Loading " + s_filename)
        shfile = HDF5File(s_filename, 'r')
        thfile = HDF5File(t_filename, 'r')
        siid = shfile["image_index"][:]
        tiid = thfile["image_index"][:]
        np.random.shuffle(siid)
        np.random.shuffle(tiid)
        trainIdx = siid[0:nTrain]
        trainTIdx =tiid[0:transferImages]
        testIdx  = tiid[transferImages:transferImages+MAX_TEST]
        train.append((ClassIndexes(s_filename, trainIdx), ClassIndexes(t_filename, trainTIdx))) #data is actually loaded only when needed
        test.append([ClassIndexes(t_filename, testIdx)])
        shfile.close()
        thfile.close()
    return Data(train, test)

def load_dataset_da(all_class_indexes, output_folder):
    for ci, tuple_c in enumerate(all_class_indexes):
        n_patches = 0
        n_images = 0
        for class_idx in tuple_c:
            n_patches += (class_idx.index[:,1]-class_idx.index[:,0]).sum()
            n_images += class_idx.index.shape[0]
        patches = np.empty([n_patches, patchOptions.patch_dim])
        positions = np.empty([n_patches,2])
        idx = np.empty([n_images, 2])
        current = 0
        current_image = 0
        get_logger().info("Loading class " + str(ci))
        for c in tuple_c:
            misc = load_patches(c)
            class_samples = misc.n_samples
            class_images = misc.indexes.shape[0]
            patches[current:current+class_samples] = misc.patches
            positions[current:current+class_samples] = misc.tags
            #import pdb; pdb.set_trace()
            idx[current_image:current_image+class_images] = misc.indexes + current
            current += class_samples
            current_image += class_images
            get_logger().info("Loaded " + str(class_samples) + "-" + str(class_images) + " for class " + c.filename)
            if patchOptions().relu:
                get_logger().info("Applying RELU")
                patches[patches < 0]=0
            make_split(tuple_c[0].filename, output_folder,patches, positions, idx)
    get_logger().info("Sizes: " + str(patches.shape) + " " + str(positions.shape))
    return MiscData(patches, positions, idx)


def make_split(output_name, output_subdir, _patches, _positions, _image_indexes):
    log = get_logger()
    #import pdb; pdb.set_trace()
    try:
        makedirs(output_subdir)
    except:
        pass
    output_filename = join(output_subdir, basename(output_name))
    log.debug('Saving extracted descriptors to %s', output_filename)


    hfile = HDF5File(output_filename, 'w', compression='gzip', fillvalue=0.0)
    patches = hfile.create_dataset('patches', _patches.shape, dtype="float32", chunks=True)
    positions = hfile.create_dataset('positions', _positions.shape, dtype="uint16", chunks=True)
    image_index = hfile.create_dataset('image_index', _image_indexes.shape, dtype='uint64') # Start, End positions of an image
    patches[:]= _patches
    positions[:]=_positions
    image_index[:]=_image_indexes
    patches.attrs['cursor'] = 0
    patches.attrs['feature_type'] = "CAFFE"
    hfile.close()

if __name__ == '__main__':
    init_logging()
    log = get_logger()
    args = get_arguments()
    source = args.source_dir
    target = args.target_dir
    output_dir = args.output_dir
    train_images = args.train_images
    transfer_imgs = args.transfer_imgs
    split = args.split
    data_indexes = get_indexes_da(source, target, train_images, transfer_imgs)
    load_dataset_da(data_indexes.train, join(output_dir,"train","split_" + str(split)))
    load_dataset_da(data_indexes.test, join(output_dir,"test","split_" + str(split)))