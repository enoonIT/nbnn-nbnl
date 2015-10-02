
import Image
from os.path import basename
import numpy as np
from itertools import product

import caffe

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

class FlipX(ImTransform):
    def apply(self, im):
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    def check_coords(self, x1, y1, x2, y2):
        return True
    def __str__(self):
        return 'flipX'

class FlipY(ImTransform):
    def apply(self, im):
        return im.transpose(Image.FLIP_TOP_BOTTOM)
    def check_coords(self, x1, y1, x2, y2):
        return True
    def __str__(self):
        return 'flipY'

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
        self.patches6 = np.zeros( (num_items, dim), dtype='float' )
        self.patches7 = np.zeros( (num_items, dim), dtype='float' )
        self.pos = np.zeros( (num_items, 2), dtype='float' )
        self.cursor = 0

    def append(self, features6, features7, pos):
        self.patches6[self.cursor:self.cursor+features6.shape[0], :] = features6
        self.patches7[self.cursor:self.cursor+features7.shape[0], :] = features7
        if (pos.shape[0] == 1) and (features6.shape[0]):
            # If there's mismatch in number of features and positions,
            # replicating positions
            pos = np.tile(pos, (features6.shape[0], pos.shape[0]) )

        self.pos[self.cursor:self.cursor+pos.shape[0], :] = pos
        self.cursor += features.shape[0]

    def get(self):
        self.patches6.resize( (self.cursor, self.patches6.shape[1]) )
        self.patches7.resize( (self.cursor, self.patches7.shape[1]) )
        self.pos.resize( (self.cursor, self.pos.shape[1]) )

        return (self.patches6, self.patches7, self.pos)

class CaffeExtractorPlus:
    def __init__(self, layer_name, model_path, meta_path, data_mean_path):
        self.layer_name = layer_name
        caffe.set_mode_cpu()
        self.net = caffe.Net(meta_path, model_path, caffe.TEST)
        self.transforms = [NopTransform()]
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(data_mean_path).mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        self.net.blobs['data'].reshape(1,3,227,227)

    def set_parameters(self, patch_size, patches_per_image, levels, image_dim, decaf_oversample, extraction_method):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.levels = levels
        self.image_dim = image_dim
        self.decaf_oversample = decaf_oversample
	self.extraction_method=self.extract
	if(extraction_method=="extra"):
	  self.extraction_method=self.balanced_extract
        self.patch_sizes = map(int, self.patch_size * 2**np.arange(0,levels,1.0))
        print self.patch_sizes

    def add_transform(self, transform):
        self.transforms.append(transform)

    def get_descriptor_size(self):
        if self.layer_name in ['67_relu', '67']:
            return 8192
        else:
            return 4096
    def to_rgb(self, _im):
        im = np.array(_im)
        if(im.ndim==3):
            ret = im/255.0
        else:
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=np.float32)
            ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] = im/255.0

        return ret

    def get_decaf(self, im):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', self.to_rgb(im))
        self.net.forward()
        features7 = self.net.blobs['fc7'].data[0]
        features6 = self.net.blobs['fc6'].data[0]
        features = features7
        if self.layer_name == '67':
            features = np.hstack([features6, features7])
        return features.reshape(1, features.shape[0])


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
            self.extraction_method(im_, feature_storage, xform.check_coords, xform, filename)

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
            #print "Level : " + str(l)
            count = 0
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
                        count += 1
                        feature_storage.append( self.get_decaf(patch), np.matrix([x, y]) )
            print "At level " + str(l) + " extracted: " + str(count)

    def balanced_extract(self, im, feature_storage, check_patch_coords, transform, filename):
        (w, h) = im.size

        # Calculating patch step
        if self.levels > 0:
            patchesXLevel = self.patches_per_image/len(self.patch_sizes)
            print "Patches per level: " + str(patchesXLevel)

        if isinstance(transform, NopTransform): # Hacky....
            # Extracting features for the whole image
            feature_storage.append( self.get_decaf(im), np.matrix([0,0]) )
        expected = 0
        skipped = 0
        # Extracting features from patches
        for l in range(self.levels):
            countLevel = 0
            _w = w -self.patch_sizes[l]
            _h = h - self.patch_sizes[l]
            if(_w <0 or _h<0):
                continue
            patch_step = int( (_w*_h / patchesXLevel)**0.52 )+2
            w_steps = np.arange(0, _w+1, patch_step)
            h_steps = np.arange(0, _h+1, patch_step)
            print "Image size (" + str(w)+", "+str(h)+") - patch size: " + str(self.patch_sizes[l]) + " patch step: " + str(patch_step) + " available pixels: (" + str(_w) +", "+str(_h)+") " \
                    "\n\twsteps: " + str(w_steps) + " \n\th_steps: " + str(h_steps)
            for i in range(len(w_steps)):
                for j in range(len(h_steps)):
                    expected += 1
                    x = w_steps[i]
                    y = h_steps[j]
                    patch_left = x+self.patch_sizes[l]
                    patch_bottom = y+self.patch_sizes[l]

                    if (check_patch_coords(x, y, patch_left, patch_bottom) and
                        patch_left <= w and patch_bottom <= h ):
                        patch = im.crop( (x, y, patch_left, patch_bottom) )
                        patch.load()
                        countLevel +=1
                        feature_storage.append( self.get_decaf(patch), np.matrix([x, y]) )
                    else:
                        skipped += 1
            print "got " + str(countLevel) + " for level " + str(l)
        print "Expected " + str(expected) + " skipped: " + str(skipped)


def get_oversampled_demo_ex():
    ex = CaffeExtractor()
    ex.set_parameters(64, 100, 3, 256)

    xforms_A = [Zoom(1), Zoom(2)]
    xforms_B = [Rotation(a) for a in range(-120,0,30) + range(30,150,30)]
    xforms = map(lambda (x,y): CombinedTransform(x,y), product(xforms_A, xforms_B))
    ex.transforms.extend(xforms)

    return ex

def get_full_image_demo_ex():
    ex = CaffeExtractor()
    ex.set_parameters(64, 1, 0, 256)

    return ex

def get_3_levels_demo_ex():
    ex = CaffeExtractor()
    ex.set_parameters(64, 100, 3, 256)

    return ex

if __name__ == '__main__':
    testImage = "/home/enoon/nbnn_cnn_dist/data/images/scene15/bedroom/image_0020.jpg"
    ex1 = CaffeExtractor("")
    ex1.set_parameters(32,100,1,200)
    ex2 = CaffeExtractor("")
    ex2.set_parameters(32,100,3,200)
    #result1 = ex1.extract_image(testImage)
    result2 = ex2.extract_image(testImage)

    #print result1.cursor
    print result2.cursor
