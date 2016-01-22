from argparse import ArgumentParser
from h5py import File as hfile
import numpy as np
import pyflann
import pickle
from PIL import Image, ImageDraw
from glob import glob
from os.path import basename
import os

classes_dir = ["RockClimbing", "badminton", "bocce", "croquet", "polo", "rowing", "sailing", "snowboarding"]


def get_arguments():
    parser = ArgumentParser(description='Draw best patches from given class')
    #parser.add_argument("selected_image", help="Filename of the image")
    #parser.add_argument("images_folder", help="Folder containing the images to parse")
    parser.add_argument("hdf_folder", help="The HDF5 file containing the class patches for the selected image")
    #parser.add_argument("second_class_hdf", help="The HDF5 file containing the class patches for a second class")
    parser.add_argument("num_patches", help="The number of patches to highlight", type=int)
    parser.add_argument("scaler", help="Location of the standard scaler")
    parser.add_argument("output_folder", help="Where to save the generated images")
    args = parser.parse_args()
    return args


def save_crop(patch_loc, image_path, num, out_folder):
    scale = 2
    image_dim = 200 * scale
    im = Image.open(image_path)
    if max(im.size) != image_dim:
        if im.size[0] > im.size[1]:
            new_height = (image_dim * im.size[1]) / im.size[0]
            new_dim = (image_dim, new_height)
        else:
            new_width = (image_dim * im.size[0]) / im.size[1]
            new_dim = (new_width, image_dim)
        print 'Resizing image from (%d, %d) to (%d, %d).' % (im.size[0], im.size[1], new_dim[0], new_dim[1])
        im = im.resize(new_dim, Image.ANTIALIAS)
    pos = patch_loc[0] * scale
    size = patch_loc[1] * scale
    endPos = pos + size
    im.crop((pos[0], pos[1], endPos[0], endPos[1])).save(out_folder + "patch_" + str(num) + ".jpg")


def get_nearest_patches(hfile_path, modelFile, scaler, n_patches, out_folder):
    classSupport = np.loadtxt(open(modelFile), skiprows=1)
    classSupport = classSupport.T.astype("float32")
    sfile = hfile(hfile_path, "r")
    im_keys = sfile["keys"][:]
    iid = sfile["image_index"][:]
    p7 = sfile["patches7"][:iid.max()].clip(0)
    pos = sfile["positions"][:iid.max()]
    p7 = np.hstack([p7, pos.copy()])
    p7 = scaler.transform(p7)
    engine = pyflann.FLANN()
    engine.build_index(classSupport, algorithm='kdtree', trees=4)
    (_, dist) = engine.nn_index(p7, num_neighbors=1)
    bestIdx = dist.argsort()[:n_patches]
    image_ids = [(idx, np.where(iid <= idx)[0][-1]) for idx in bestIdx if idx != 0]
    sizes = [32, 64, 128]
    for num, idx in enumerate(image_ids):
        image_id = idx[1]  # actual image id
        patch_n = idx[0] - iid[image_id, 0]  # patch number relative to image
        image_pos = pos[iid[image_id, 0]:iid[image_id, 1]]
        zeroLoc = np.where(np.all(image_pos == [0, 0], axis=1))[0][1:]
        patch_loc = (image_pos[patch_n, :], sizes[np.where(zeroLoc <= patch_n)[0][-1]])  # pos and size of the patch
        save_crop(patch_loc, im_keys[image_id], num, out_folder)
    #import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = get_arguments()
    scaler = pickle.load(open(args.scaler))
    for idx, cl in enumerate(classes_dir):
        hdf_file = args.hdf_folder + "/" + cl + ".hdf5"
        #image_folder = args.images_folder + "/" + cl + "/"
        modelFile = "class" + str(idx) + "W.txt"
        out_folder = args.output_folder + "/" + cl + "/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        patches = get_nearest_patches(hdf_file, modelFile, scaler, args.num_patches, out_folder)
        #break
