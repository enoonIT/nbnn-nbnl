from argparse import ArgumentParser
import numpy as np
from h5py import File as hfile
from PIL import Image, ImageDraw, ImageChops
import os

class_names = ["RockClimbing", "badminton", "bocce", "croquet", "polo", "rowing", "sailing", "snowboarding"]
#class_names = ["CALsuburb", "MITcoast", "MITforest", "MIThighway", "MITinsidecity", "MITmountain", "MITopencountry",
#"MITstreet", "MITtallbuilding", "PARoffice", "bedroom", "industrial", "kitchen", "livingroom", "store"]


def get_arguments():
    parser = ArgumentParser(description='Draw best patches from given class')
    parser.add_argument("score_file", help="")
    parser.add_argument("hdf_folder", help="")
    parser.add_argument("num_patches", help="The number of patches to highlight", type=int)
    parser.add_argument("output_folder", help="Where to save the generated images")
    args = parser.parse_args()
    return args


def get_best_patches_idx(score_file):
    data = open(score_file).readlines()
    best_scores = []
    for N in range(0, len(class_names)):
        classN = [l[1:].strip().split()[N] for l in data if l.split()[0] == str(N)]
        patch_scores = np.asarray(classN).astype("float32")
        print "We have " + str(patch_scores.shape) + " scores for class " + str(N)
        best = patch_scores.argsort()[::-1][:args.num_patches]
        best_scores.append(best)
        print "\tBest score is %f, worst score (of the best) is %f" % (patch_scores[best[0]], patch_scores[best[-1]])
    return best_scores


def get_size(image_pos, patch_id):
    #import pdb; pdb.set_trace()
    if(int(patch_id) == 0):
        return None
    return 32
    sizes = [32, 64, 128]
    zeroLoc = np.where(np.all(image_pos == [0, 0], axis=1))[0][1:]
    return sizes[np.where(zeroLoc <= patch_id)[0][-1]]


def save_crop(pos, size, image_path, num, out_folder):
    scale = 1
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
    pos = pos * scale
    if size is None:
        size = image_dim
    else:
        size = size * scale
    endPos = pos + size
    im.crop((pos[0], pos[1], endPos[0], endPos[1])).resize([200,200], Image.ANTIALIAS).save(out_folder + "patch_" + str(num) + ".jpg")


def show_bounding_boxes(image, patches, color, output_filename):
    scale = 4
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


def show_heat_map(image, patches, color, output_filename):
    scale = 4
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
    base_c = 40
    heatmap = Image.new("RGB", im.size, (base_c, base_c, base_c))
    draw = ImageDraw.Draw(heatmap, "RGBA")
    coeff = 2 * (255 - base_c) / len(patches)
    coeff = max(coeff, 160)
    for patch in patches:
        pos = patch[0] * scale
        size = patch[1] * scale
        endPos = pos + size
        draw.rectangle([pos[0], pos[1], endPos[0], endPos[1]], fill=(255, 255, 255, coeff))
        #draw.rectangle([pos[0], pos[1], endPos[0], endPos[1]], outline=color)
    #im.show()
    ImageChops.multiply(im, heatmap).save(output_filename, "jpeg")
    print "Saved image " + output_filename


def show_patches(hdf_file, patch_idx, output_folder):
    print "Loading " + hdf_file
    class_hdf = hfile(hdf_file, "r")
    iid = class_hdf["image_index"][:]
    names = class_hdf["keys"][:]
    positions = class_hdf["positions"][:]
    images_to_patches = {}
    for n, patch_id in enumerate(patch_idx):
        assert(patch_id < iid.max())
        image_id = np.where(iid <= patch_id)[0][-1]
        print "Patch %d matches image %d - %s, range %s" % (patch_id, image_id, names[image_id], iid[image_id])
        rel_patch_id = patch_id - iid[image_id][0]
        rel_pos = positions[iid[image_id][0]:iid[image_id][1]]
        patch_pos = positions[patch_id]
        assert(np.all(rel_pos[rel_patch_id] == patch_pos))
        size = get_size(rel_pos, rel_patch_id)
        if size is None:
            continue
        im_patches = images_to_patches.get(names[image_id], [])
        im_patches.append((patch_pos, size))
        images_to_patches[names[image_id]] = im_patches
        #save_crop(patch_pos, size, names[image_id], n, output_folder)
    for image in images_to_patches.keys():
        if(len(images_to_patches[image]) < 3):
            continue
        show_heat_map(Image.open(image), images_to_patches[image], "#ff0000", output_folder + os.path.basename(image))
        #show_bounding_boxes(Image.open(image), images_to_patches[image], "#ff0000", output_folder + os.path.basename(image))


if __name__ == "__main__":
    args = get_arguments()
    best_patch_index = get_best_patches_idx(args.score_file)
    for c_id, class_name in enumerate(class_names):
        class_hdf = args.hdf_folder + "/" + class_name + ".hdf5"
        output_folder = args.output_folder + "/" + class_name + "/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        show_patches(class_hdf, best_patch_index[c_id], output_folder)

