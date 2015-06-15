
import Image
from os.path import basename
import numpy as np
from itertools import product

from decaf.scripts.imagenet import DecafNet

from common import get_logger, init_logging
from rotation_helpers import rect_in_rotated_rect

class ImTransform:
    def apply(self, im):
        pass
    def check_coords(self, x1, y1, x2, y2):
        pass
    def __str__(self):
        pass

class NopTransform(ImTransform):
    def apply(self, im):
        return im
    def check_coords(self, x1, y1, x2, y2):
        return True
    def __str__(self):
        return 'nop'

class Rotation(ImTransform):
    def __init__(self, angle):
        self.angle = angle

    def apply(self, im):
        (w_orig, h_orig) = im.size

        im_ = im.rotate(self.angle, Image.BILINEAR, expand=True)

        (w, h) = im_.size
        self.orig_im_coords = [w/2 - w_orig/2, h/2 - h_orig/2,
                               w/2 + w_orig/2, h/2 + h_orig/2]
        return im_

    def check_coords(self, x1, y1, x2, y2):
        return rect_in_rotated_rect([x1,y1,x2,y2], self.orig_im_coords,
                                    self.angle)

    def __str__(self):
        return 'rotation_%d' % self.angle

class Zoom(ImTransform):
    def __init__(self, times):
        self.times = times
    def apply(self, im):
        return im.resize( (im.size[0] * self.times, im.size[1] * self.times) )
    def check_coords(self, *params):
        return True
    def __str__(self):
        return 'zoom_%d' % self.times

class CombinedTransform(ImTransform):
    def __init__(self, xform_A, xform_B):
        self.xform_A = xform_A
        self.xform_B = xform_B

    def apply(self, im):
        im_ = self.xform_A.apply(im)
        return self.xform_B.apply(im_)

    def check_coords(self, x1, y1, x2, y2):
        return self.xform_A.check_coords(x1, y1, x2, y2) and \
            self.xform_B.check_coords(x1, y1, x2, y2)

    def __str__(self):
        return '%s_>_%s' % (self.xform_A, self.xform_B)

class ExtractedFeatures:
    def __init__(self, num_items, dim):
        self.patches = np.zeros( (num_items, dim), dtype='float' )
        self.pos = np.zeros( (num_items, 2), dtype='float' )
        self.cursor = 0

    def append(self, features, pos):
        self.patches[self.cursor:self.cursor+features.shape[0], :] = features
        if (pos.shape[0] == 1) and (features.shape[0]):
            # If there's mismatch in number of features and positions,
            # replicating positions
            pos = np.tile(pos, (features.shape[0], pos.shape[0]) )

        self.pos[self.cursor:self.cursor+pos.shape[0], :] = pos
        self.cursor += features.shape[0]

    def get(self):
        self.patches.resize( (self.cursor, self.patches.shape[1]) )
        self.pos.resize( (self.cursor, self.pos.shape[1]) )

        return (self.patches, self.pos)

class DecafExtractor:
    def __init__(self, layer_name,
                 model_path = 'dist/decaf-release/model/imagenet.decafnet.epoch90',
                 meta_path = 'dist/decaf-release/model/imagenet.decafnet.meta'):
        self.layer_name = layer_name
        self.net = DecafNet(model_path, meta_path)
        self.transforms = [NopTransform()]

    def set_parameters(self, patch_size, patches_per_image, levels, image_dim, decaf_oversample=False):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.levels = levels
        self.image_dim = image_dim
        self.decaf_oversample = decaf_oversample

        self.patch_sizes = map(int, self.patch_size * 2**np.arange(0,levels,1.0))

    def add_transform(self, transform):
        self.transforms.append(transform)

    def get_descriptor_size(self):
        if self.layer_name in ['67_relu', '67']:
            return 8192
        else:
            return 4096

    def get_decaf(self, im):
        scores = self.net.classify(np.asarray(im), center_only=not self.decaf_oversample)
        if self.layer_name == '67_relu':
            return np.hstack([self.net.feature('fc6_neuron_cudanet_out'),
                              self.net.feature('fc7_neuron_cudanet_out')])
        elif self.layer_name == '67':
            return np.hstack([self.net.feature('fc6_cudanet_out'),
                              self.net.feature('fc7_cudanet_out')])
        else:
            return self.net.feature(self.layer_name)


    def get_number_of_features_per_image(self):
        if self.decaf_oversample:
            return 10*len(self.transforms)
        else:
            return len(self.transforms)

    def extract_image(self, filename):
        """ This method extracts 4096-dimensional DeCAF features from
        patches at multiple scales belonging to the the image file <filename>.
        Number of scales, patch size, and other settings are set by method
        set_parameters().

        This method returns tuple (patches, positions, patch_number),
        where <patches> is a numpy array of dimension (patch_number, 4096) holding features,
        <positions> is a numpy array of dimension (patch_number, 2) holding absolute positions of patches,
        <patch_number> is the number of patches that algorithm managed to extract -- it
        is guaranteed to be at most the number of originally specified.
        """
        log = get_logger()

        im = Image.open(filename)

        # Resizing image
        if max(im.size) != self.image_dim:
            if im.size[0] > im.size[1]:
                new_height = (self.image_dim * im.size[1]) / im.size[0]
                new_dim = (self.image_dim, new_height)
            else:
                new_width = (self.image_dim * im.size[0]) / im.size[1]
                new_dim = (new_width, self.image_dim)

            log.info('Resizing image from (%d, %d) to (%d, %d).', im.size[0], im.size[1], new_dim[0], new_dim[1])
            im = im.resize(new_dim, Image.ANTIALIAS)

        # Estimating number of extracted features taking into account transformations
        estimated_feature_num = self.patches_per_image * self.get_number_of_features_per_image()

        # Provisioning space for patches and locations
        feature_storage = ExtractedFeatures(estimated_feature_num, self.get_descriptor_size())

        log.info('Extracting up to %d patches at %d levels from "%s"...',
                 self.patches_per_image * self.get_number_of_features_per_image(),
                 self.levels, basename(filename))

        # Applying transformations and extracting features
        for xform in self.transforms:
            im_ = xform.apply(im)
            self.extract(im_, feature_storage, xform.check_coords, xform, filename)

        log.info('Done. Extracted: %d.', feature_storage.cursor)
        return feature_storage


    def extract(self, im, feature_storage, check_patch_coords, transform, filename):
        (w, h) = im.size

        # Calculating patch step
        if self.levels > 0:
            patch_step = int( (w*h * len(self.patch_sizes) / self.patches_per_image)**0.5 )
            w_steps = np.arange(0, w, patch_step)
            h_steps = np.arange(0, h, patch_step)
            (xx, yy) = np.meshgrid(w_steps, h_steps)

        if isinstance(transform, NopTransform): # Hacky....
            # Extracting features for the whole image
            feature_storage.append( self.get_decaf(im), np.matrix([0,0]) )

        # Extracting features from patches
        for l in range(self.levels):
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    x = xx[i,j]
                    y = yy[i,j]
                    patch_left = x+self.patch_sizes[l]
                    patch_bottom = y+self.patch_sizes[l]

                    if (check_patch_coords(x, y, patch_left, patch_bottom) and
                        patch_left <= w and patch_bottom <= h ):
                        patch = im.crop( (x, y, patch_left, patch_bottom) )
                        patch.load()

                        feature_storage.append( self.get_decaf(patch), np.matrix([x, y]) )


def get_oversampled_demo_ex():
    ex = DecafExtractor()
    ex.set_parameters(64, 100, 3, 256)

    xforms_A = [Zoom(1), Zoom(2)]
    xforms_B = [Rotation(a) for a in range(-120,0,30) + range(30,150,30)]
    xforms = map(lambda (x,y): CombinedTransform(x,y), product(xforms_A, xforms_B))
    ex.transforms.extend(xforms)

    return ex

def get_full_image_demo_ex():
    ex = DecafExtractor()
    ex.set_parameters(64, 1, 0, 256)

    return ex

def get_3_levels_demo_ex():
    ex = DecafExtractor()
    ex.set_parameters(64, 100, 3, 256)

    return ex


