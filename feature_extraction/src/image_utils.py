from PIL import Image

from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser(description='Utility to resize images while keeping aspect ratio')
    parser.add_argument("image_path", help="Path to image")
    parser.add_argument("square_side", help="Size of square side", type=int)
    args = parser.parse_args()
    return args


def scale_image_keep_ratio(image, square_side):
    image.show()
    largest_side = max(image.size)
    ratio = square_side / float(largest_side)
    new_image = Image.new(image.mode, (square_side, square_side))
    print "Old size " + str(image.size) + " new size " + str(new_image.size)
    new_w = int(image.width * ratio)
    new_h = int(image.height * ratio)
    #import pdb; pdb.set_trace()
    scaled_image = image.resize((new_w, new_h))
    if (new_w == new_h == square_side):
        return scaled_image
    top_h = (square_side - new_h) / 2
    bottom_h = square_side - (new_h + top_h)
    left_w = (square_side - new_w) / 2
    right_w = square_side - (new_w + left_w)
    new_image.paste(scaled_image, (left_w, top_h))
    if(new_w > new_h):  # empty vertical space
        top = scaled_image.transform((square_side, top_h), Image.EXTENT, (0, 0, new_w, 1))
        bottom = scaled_image.transform((square_side, bottom_h), Image.EXTENT, (0, new_h - 1, new_w, new_h))
        new_image.paste(top, (0, 0))
        new_image.paste(bottom, (0, top_h + new_h))
    else:  # empty horizontal space
        left = scaled_image.transform((left_w, square_side), Image.EXTENT, (0, 0, 1, new_h))
        right = scaled_image.transform((right_w, square_side), Image.EXTENT, (new_w - 1, 0, new_w, new_h))
        new_image.paste(left, (0, 0))
        new_image.paste(right, (left_w + new_w, 0))
    return new_image


if __name__ == '__main__':
    args = get_arguments()
    img = scale_image_keep_ratio(Image.open(args.image_path, "r"), args.square_side)
    img.show()