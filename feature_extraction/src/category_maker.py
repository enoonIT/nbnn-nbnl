from argparse import ArgumentParser
from h5py import File as hfile
import numpy as np
import pyflann
from PIL import Image
from glob import glob
from os.path import basename
import os
from shutil import copyfile

def get_arguments():
    parser = ArgumentParser(description='Draw best patches from given class')
    parser.add_argument("input_file", help="Filename of the image")
    parser.add_argument("image_folder", help="Where to take the images from")
    parser.add_argument("output_folder", help="Where to put the images divided by category")
    args = parser.parse_args()
    return args


def parse_file_VOC(input_file, image_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(input_file) as cat_file:
        n_files = 0
        for line in cat_file.readlines():
            if line.split()[1] == "1":
                n_files += 1
                image_name = "/" + line.split()[0] + ".jpg"
                copyfile(image_path + image_name, output_path + image_name)

        print "Found " + str(n_files) + " valid files"


def parse_file_art(input_file, image_path, output_path):
    with open(input_file) as cat_file:
        for line in cat_file.readlines()[1:]:
            #import pdb; pdb.set_trace()
            cells = line.split(";")
            cells = [cell.replace('\n', '').strip("'") for cell in cells[1:]]
            imagename = "/" + basename(cells[0])
            trainTest = cells[2]
            categories = cells[3]
            source_image = image_path + imagename
            print "FROM " + source_image
            for categ in categories.split():
                dest = output_path + "/" + trainTest + "/" + categ
                if not os.path.exists(dest):
                    os.makedirs(dest)
                print "\t" + dest + imagename
                if os.path.exists(source_image):
                    copyfile(source_image, dest + imagename)
                else:
                    print "Source image missing"


if __name__ == '__main__':
    args = get_arguments()
    parse_file_art(args.input_file, args.image_folder, args.output_folder)
    #postfix = "_trainval.txt"
    #cat_files = sorted(glob(args.input_folder + "/*" + postfix))
    #for catf in cat_files:
        #category = basename(catf[:-len(postfix)])
        #print category
        #parse_file(catf, args.image_folder, args.output_folder + "/" + category)

