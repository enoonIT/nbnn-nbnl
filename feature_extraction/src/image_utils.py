from PIL import Image
from os import path, makedirs
from argparse import ArgumentParser
import cv2


def get_arguments():
    parser = ArgumentParser(description='Utility to resize images while keeping aspect ratio')
    #parser.add_argument("image_path", help="Path to image")
    parser.add_argument("input_folder", help="From where to load the scaled images")
    parser.add_argument("files_to_scale", help="The file containing the names of the images we want to scale")
    parser.add_argument("output_folder", help="Where to scale the scaled images")
    parser.add_argument("square_side", help="Size of square side", type=int)
    parser.add_argument("--depth2rgb", help="If this flag is enabled, images will be converted to RGB with a "
                        "jet colorization", action='store_true')
    args = parser.parse_args()
    return args


def scale_image_keep_ratio(image, square_side):
    #image.show()
    largest_side = max(image.size)
    ratio = square_side / float(largest_side)
    new_image = Image.new(image.mode, (square_side, square_side))
    #print "Old size " + str(image.size) + " new size " + str(new_image.size)
    new_w = int(round(image.width * ratio))
    new_h = int(round(image.height * ratio))
    #import pdb; pdb.set_trace()
    scaled_image = image.resize((new_w, new_h))
    if (new_w == new_h == square_side):
        return scaled_image
    top_h = (square_side - new_h) / 2
    bottom_h = square_side - (new_h + top_h)
    left_w = (square_side - new_w) / 2
    right_w = square_side - (new_w + left_w)
    #print "Paddings " + str((top_h, bottom_h, left_w, right_w))
    new_image.paste(scaled_image, (left_w, top_h))
    if(new_w > new_h) and (top_h + bottom_h) > 1:  # empty vertical space
        top = scaled_image.transform((square_side, top_h), Image.EXTENT, (0, 0, new_w, 1))
        bottom = scaled_image.transform((square_side, bottom_h), Image.EXTENT, (0, new_h - 1, new_w, new_h))
        new_image.paste(top, (0, 0))
        new_image.paste(bottom, (0, top_h + new_h))
    elif (left_w + right_w) > 1:  # empty horizontal space
        left = scaled_image.transform((left_w, square_side), Image.EXTENT, (0, 0, 1, new_h))
        right = scaled_image.transform((right_w, square_side), Image.EXTENT, (new_w - 1, 0, new_w, new_h))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (left_w + new_w, 0))
    return new_image


def convert_depth_image_to_jetrgb(image_path):
    dimage = cv2.imread(image_path, -1)  # load in raw mode
    min_value = dimage.min()
    max_value = dimage.max()
    im_range = (dimage.astype('float32') - min_value) / (max_value - min_value)
    im_range = 255.0 * im_range
    out_img = cv2.applyColorMap(im_range.astype("uint8"), cv2.COLORMAP_JET)
    return Image.fromarray(out_img)


def scale_images(input_folder, output_folder, filenames, new_side_size, depth2rgb):
    total_images = len(filenames)
    print "There are " + str(total_images) + " to scale"
    for item, file_line in enumerate(filenames):
        rel_path = file_line.split()[0]
        input_path = path.join(input_folder, rel_path)
        output_path = path.join(output_folder, rel_path)
        if(path.exists(output_path)):
            continue
        if depth2rgb:
            image = convert_depth_image_to_jetrgb(input_path)
        else:
            image = Image.open(input_path, "r")
        new_image = scale_image_keep_ratio(image, new_side_size)
        folder_structure = path.dirname(output_path)
        if not path.exists(folder_structure):
            makedirs(folder_structure)
        new_image.save(output_path)
        if (item % 1000) == 0:
            print str(item) + " out of " + str(total_images)


if __name__ == '__main__':
    args = get_arguments()
    with open(args.files_to_scale) as eval_file:
        scale_images(args.input_folder, args.output_folder, eval_file.readlines(), args.square_side, args.depth2rgb)