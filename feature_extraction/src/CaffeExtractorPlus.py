
from PIL import Image
from os.path import basename
import numpy as np
from itertools import product
import ExtractedFeatures
import CaffeExtractor
from common import get_logger, init_logging
import caffe, time

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

class CaffeExtractorPlus:
    def __init__(self, model_path, meta_path, data_mean_path):
        caffe.set_mode_gpu()
        self.net = caffe.Net(meta_path, model_path, caffe.TEST)
        self.transforms = [NopTransform()]
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', np.load(data_mean_path).mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    def set_parameters(self, patch_size, patches_per_image, levels, image_dim, batch_size):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.levels = levels
        self.image_dim = image_dim
        self.extraction_method=self.balanced_extract
        self.patch_sizes = map(int, self.patch_size * 2**np.arange(0,levels,1.0))
        self.batch_size = batch_size;
        self.net.blobs['data'].reshape(self.batch_size,3,227,227)
        print self.patch_sizes

    def add_transform(self, transform):
        self.transforms.append(transform)
    def enable_data_augmentation(self):
        self.transforms.append(FlipX())
        self.transforms.append(FlipY())
        self.transforms.append(CombinedTransform(FlipX(), FlipY()))
        self.transforms.append(CombinedTransform(FlipY(), FlipX()))

    def get_number_of_features_per_image(self):
        return len(self.transforms)

    def get_descriptor_size(self):
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

    def load_caffe_patches(self, preprocessed_patches, positions, extracted_features):
        nPatches = preprocessed_patches.shape[0]
        start = 0
        print "Will now extract patches, we have " + str(preprocessed_patches.shape) + " patches"
        while nPatches > 0:
            extractedPatches = min(nPatches, self.batch_size)
            print "Extracting " + str(extractedPatches)
            self.net.blobs['data'].data[0:extractedPatches] = preprocessed_patches[start:start+extractedPatches]
            self.net.forward()
            features7 = self.net.blobs['fc7'].data[0:extractedPatches]
            features6 = self.net.blobs['fc6'].data[0:extractedPatches]
            extracted_features.append(features6, features7, positions[start:start+extractedPatches])
            #prepare for next batch
            nPatches -= self.batch_size
            start += self.batch_size

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
        feature_storage = ExtractedFeatures.ExtractedFeatures(estimated_feature_num, self.get_descriptor_size())

        log.info('Extracting up to %d patches at %d levels from "%s"...',
                 self.patches_per_image * self.get_number_of_features_per_image(),
                 self.levels, basename(filename))

        # Applying transformations and extracting features
        for xform in self.transforms:
            im_ = xform.apply(im)
            self.extraction_method(im_, feature_storage, xform.check_coords, xform, filename)

        log.info('Done. Extracted: %d.', feature_storage.cursor)
        return feature_storage

    def balanced_extract(self, im, feature_storage, check_patch_coords, transform, filename):
        (w, h) = im.size
        # Extracting features from patches
        preprocessedPatches = np.empty([self.patches_per_image, 3, 227, 227], dtype="float32")
        positions = np.zeros((self.patches_per_image,2), dtype="uint16")
        # Calculating patch step
        if self.levels > 0:
            patchesXLevel = self.patches_per_image/len(self.patch_sizes)
            print "Patches per level: " + str(patchesXLevel)
        k=0
        if isinstance(transform, NopTransform): # Hacky.... #TODO why only for NopTransform?
            # Extracting features for the whole image
            preprocessedPatches[k,...]=self.transformer.preprocess('data', self.to_rgb(im))
            positions[k,...]= np.matrix([0,0])
            k+=1
        expected = 0
        skipped = 0

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
                        preprocessedPatches[k,...]=self.transformer.preprocess('data', self.to_rgb(patch))
                        positions[k,...] = np.matrix([x, y])
                        k+=1
                    else:
                        skipped += 1
            print "got " + str(countLevel) + " for level " + str(l)
        self.load_caffe_patches(preprocessedPatches[0:k], positions[0:k] ,feature_storage)
        print "Expected " + str(expected) + " skipped: " + str(skipped)

def test_extraction(filename):
    cep = CaffeExtractorPlus("/home/fmcarlucci/data/network/hybridCNN_iter_700000_upgraded.caffemodel",
    "/home/fmcarlucci/data/network/hybridCNN_deploy_no_relu_upgraded.prototxt","/home/fmcarlucci/data/network/hybrid_mean.npy")
    cep.set_parameters(32,100,3,200, 20)
    cep.add_transform(FlipX())
    cep.add_transform(FlipY())
    cep.add_transform(CombinedTransform(FlipX(), FlipY()))
    return cep.extract_image(filename)

def test_extraction_speed(filename):
    cep = CaffeExtractorPlus("/home/fmcarlucci/data/network/hybridCNN_iter_700000_upgraded.caffemodel",
    "/home/fmcarlucci/data/network/hybridCNN_deploy_no_relu_upgraded.prototxt","/home/fmcarlucci/data/network/hybrid_mean.npy")
    timeResults = np.zeros((20));
    for k in range(0,20):
        cep.set_parameters(16,100,1,200, k*5+1)
        start = time.clock()
        cep.extract_image(filename)
        cep.extract_image(filename)
        cep.extract_image(filename)
        cep.extract_image(filename)
        end = time.clock()
        timeResults[k]= end-start;
    return timeResults

def comparison(filename):
    model = "/home/fmcarlucci/data/network/hybridCNN_iter_700000_upgraded.caffemodel"
    proto = "/home/fmcarlucci/data/network/hybridCNN_deploy_no_relu_upgraded.prototxt"
    meanPy = "/home/fmcarlucci/data/network/hybrid_mean.npy"
    patch = 16
    levels = 3
    patchesPerImage=100
    imageW = 200
    batchSize = 16
    cep = CaffeExtractorPlus(model, proto, meanPy)
    ce = CaffeExtractor.CaffeExtractor("67", model, proto, meanPy)
    ce.set_parameters(patch, patchesPerImage, levels, imageW, False, "extra")
    cep.set_parameters(patch,patchesPerImage,levels,imageW, batchSize)
    start = time.clock()
    for x in range(0,10):
        patches = cep.extract_image(filename)
    end = time.clock()
    batch_time = end-start
    start = time.clock()
    for x in range(0,10):
        patches2 = ce.extract_image(filename)
    end = time.clock()
    print "Batch version took " + str(batch_time)
    print "Base version took " + str(end-start)
    return [patches, patches2]

if __name__ == '__main__':
    testImage = "/home/fmcarlucci/nbnn-nbnl/feature_extraction/src/test.jpg"
    patches = test_extraction(testImage)