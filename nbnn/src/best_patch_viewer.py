from argparse import ArgumentParser
from h5py import File as hfile
import numpy as np
import pyflann
from PIL import Image, ImageDraw
from glob import glob
from os.path import basename
import os

def get_arguments():
    parser = ArgumentParser(description='Draw best patches from given class')
    #parser.add_argument("selected_image", help="Filename of the image")
    parser.add_argument("images_folder", help="Folder containing the images to parse")
    parser.add_argument("image_class_hdf", help="The HDF5 file containing the class patches for the selected image")
    #parser.add_argument("second_class_hdf", help="The HDF5 file containing the class patches for a second class")
    parser.add_argument("num_patches", help="The number of patches to highlight", type=int)
    parser.add_argument("output_folder", help="Where to save the generated images")
    args = parser.parse_args()
    return args


def load_image(image_path):
    im = Image.open(image_path)
    return im


def get_nearest_patches(support, image_patches, image_positions, n_patches):
    engine = pyflann.FLANN()
    engine.build_index(support, algorithm='kdtree', trees=4)
    (_, dist) = engine.nn_index(image_patches, num_neighbors=1)
    #import pdb; pdb.set_trace()
    bestIdx = dist.argsort()[:n_patches]
    worstIdx = dist.argsort()[-n_patches:]
    sizes = [32, 64, 128]
    zeroLoc = np.where(np.all(image_positions == [0, 0], axis=1))[0][1:]
    best_patches = [(image_positions[idx, :], sizes[np.where(zeroLoc <= idx)[0][-1]]) for idx in bestIdx if idx != 0]
    worst_patches = [(image_positions[idx, :], sizes[np.where(zeroLoc <= idx)[0][-1]]) for idx in worstIdx if idx != 0]
    return (best_patches, worst_patches)


def show_patches(image, patches, color, output_filename):
    scale = 2
    image_dim = 200 * scale
    im = image.convert("RGB")
    if max(im.size) != image_dim:
        if im.size[0] > im.size[1]:
            new_height = (image_dim * im.size[1]) / im.size[0]
            new_dim = (image_dim, new_height)
        else:
            new_width = (image_dim * im.size[0]) / im.size[1]
            new_dim = (new_width, image_dim)
        print 'Resizing image from (%d, %d) to (%d, %d).' % (im.size[0], im.size[1], new_dim[0], new_dim[1])
        im = im.resize(new_dim, Image.ANTIALIAS)
    draw = ImageDraw.Draw(im)
    for patch in patches:
        pos = patch[0] * scale
        size = patch[1] * scale
        endPos = pos + size
        draw.rectangle([pos[0], pos[1], endPos[0], endPos[1]], outline=color)
    #im.show()
    im.save(output_filename, "jpeg")

if __name__ == '__main__':
    args = get_arguments()
    sfile = hfile(args.image_class_hdf, "r")
    categ_name = basename(args.image_class_hdf).replace(".hdf5","")
    #image_path = args.selected_image
    im_keys = sfile["keys"]
    iid = sfile["image_index"]
    p7 = sfile["patches7"]
    pos = sfile["positions"]
    #second_class = hfile(args.second_class_hdf, "r")
    #p27 = second_class["patches7"]
    images_f = glob(args.images_folder + "/*.jpg")
    im_keys = [key_name.replace("//", "/") for key_name in im_keys[:]]
    out_folder = args.output_folder + "/" + categ_name + "/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for image_path in images_f:
        print "Processing image " + image_path
        image_idx = im_keys.index(image_path)
        im_idx = iid[image_idx]
        support = np.delete(p7, range(im_idx[0], im_idx[1]), axis=0)
        image_patches = p7[im_idx[0]:im_idx[1], :]
        image_pos = pos[im_idx[0]:im_idx[1], :]
        patches = get_nearest_patches(support, image_patches, image_pos, args.num_patches)
        the_image = Image.open(image_path)
        out_f = out_folder + basename(image_path)
        show_patches(the_image, patches[0], "#ff0000", out_f)
    #show_patches(the_image, patches[1], "#0000ff")
    #patches2 = get_nearest_patches(p27, image_patches, image_pos, args.num_patches)
    #show_patches(the_image, patches2[0], "#ccaa00")
    #show_patches(the_image, patches2[1], "#00aacc")
