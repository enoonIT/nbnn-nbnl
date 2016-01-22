from argparse import ArgumentParser
import numpy as np
from h5py import File as hfile
from PIL import Image

class_names = ["RockClimbing", "badminton", "bocce", "croquet", "polo", "rowing", "sailing", "snowboarding"]


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
        classN = [l[1:].strip().split()[N] for l in data if l[0] == str(N)]
        patch_scores = np.asarray(classN).astype("float32")
        print "We have " + str(patch_scores.shape) + " scores for class " + str(N)
        best = patch_scores.argsort()[::-1][:args.num_patches]
        best_scores.append(best)
        print "\tBest score is %f, worst score (of the best) is %f" % (patch_scores[best[0]], patch_scores[best[-1]])
        break
    return best_scores


def get_size(image_pos, patch_id):
    sizes = [32, 64, 128]
    zeroLoc = np.where(np.all(image_pos == [0, 0], axis=1))[0][1:]
    return sizes[np.where(zeroLoc <= patch_id)[0][-1]]


def save_crop(patch_loc, image_path, num, out_folder):
    img = Image.open(image_path)
    pos = patch_loc[0]
    size = patch_loc[1]
    endPos = pos + size
    img.crop((pos[0], pos[1], endPos[0], endPos[1])).save(out_folder + "patch_" + str(num) + ".jpg")


def show_patches(hdf_file, patch_idx, output_folder):
    print "Loading " + hdf_file
    class_hdf = hfile(hdf_file, "r")
    iid = class_hdf["image_index"][:]
    names = class_hdf["keys"][:]
    #import pdb; pdb.set_trace()
    for patch_id in patch_idx:
        assert(patch_id < iid.max())
        image_id = np.where(iid <= patch_id)[0][-1]
        print "Patch %d matches image %d - %s, range %s" % (patch_id, image_id, names[image_id], iid[image_id])


if __name__ == "__main__":
    args = get_arguments()
    best_patch_index = get_best_patches_idx(args.score_file)
    for c_id, class_name in enumerate(class_names):
        class_hdf = args.hdf_folder + "/" + class_name + ".hdf5"
        output_folder = args.output_folder + "/" + class_name + "/"
        show_patches(class_hdf, best_patch_index[c_id], output_folder)

