
from SmoothML3 import SmoothML3
import sklearn
import random
import copy
import os
import re
import pickle

from collections import namedtuple
from common import *
from data import ILSVRCPatchMinibatchLoader, MemoryLoader
from h5py import File as HDF5File

class patchOptions(object):
    patch_name="patches"
    position_name = "positions"
    relu = True
    patch_dim=4096

MiscData = namedtuple("MiscData","patches tags labels")
ClassData = namedtuple("ClassData","patches positions tags n_samples")
ClassIndexes = namedtuple("ClassIndexes", "filename index")
Data = namedtuple("Data","train test")

def load_patches(class_data):
        hfile = HDF5File(class_data.filename, 'r')
        patches = hfile[patchOptions.patch_name][:]
        positions = hfile[patchOptions.position_name][:]
        feature_dim = patchOptions.patch_dim
        indexes = class_data.index
        num_patches=(indexes[:,1]-indexes[:,0]).sum()
        loaded_patches = np.empty([num_patches, feature_dim])
        loaded_positions = np.empty([num_patches, 2])
        tags = np.zeros([num_patches,1])
        patch_start = n_image = 0
        for iid in indexes:
            n_patches = iid[1]-iid[0]
            loaded_patches[patch_start:patch_start+n_patches,:] = patches[iid[0]:iid[1],4096:]
            loaded_positions[patch_start:patch_start+n_patches,:] = positions[iid[0]:iid[1],:]
            tags[patch_start] = 1
            patch_start += n_patches
            n_image += 1
        hfile.close()
        get_logger().info("Loaded " + str(num_patches) + " patches")
        return ClassData(loaded_patches, loaded_positions, tags, num_patches)

def get_indexes(patch_folder, nTrain, nTest):
    files = sorted(glob( join(patch_folder, '*.hdf5') ), key=basename)
    train = []
    test = []
    for (classNumber,filename) in enumerate(files):
        #support_filename = join(".", basename(filename))
        hfile = HDF5File(filename, 'r')
        iid = hfile["image_index"][:]
        nImages = iid.shape[0]
        assert nImages >= (nTrain + nTest), "Not enough images! Have " + str(nImages)  + " asking for " + str(nTrain+nTest)
        np.random.shuffle(iid)
        trainIdx = iid[0:nTrain]
        testIdx  = iid[nTrain:nTrain+nTest]
        train.append(ClassIndexes(filename, trainIdx)) #data is actually loaded only when needed
        test.append(ClassIndexes(filename, testIdx))
        hfile.close()
    return Data(train, test)

def load_dataset(all_class_indexes):
    #loaded = [load_patches(c) for c in all_class_indexes] #this actually loads data in memory
    n_patches = 0
    for class_idx in all_class_indexes:
        n_patches += (class_idx.index[:,1]-class_idx.index[:,0]).sum()
    patches = np.empty([n_patches, patchOptions.patch_dim + 2])
    tags = np.empty([n_patches,1])
    labels = np.empty([n_patches,1])
    current = 0
    for ci, c in enumerate(all_class_indexes):
        misc = load_patches(c)
        class_samples = misc.n_samples
        patches[current:current+class_samples] = np.hstack([misc.patches, misc.positions])
        tags[current:current+class_samples] = misc.tags
        labels[current:current+class_samples] = ci * np.ones([class_samples, 1])
        current += class_samples
    if patchOptions().relu:
        get_logger().info("Applying RELU")
        patches[patches < 0]=0
    #positions = np.vstack([data.positions for data in loaded])
    #tags = np.vstack([data.tags for data in loaded])
    #labels = np.vstack([c*np.ones([loaded[c].patches.shape[0],1]) for c in range(len(all_class_indexes))])
    get_logger().info("Sizes: " + str(patches.shape) + " "  + str(tags.shape) + " " + str(labels.shape))
    return MiscData(patches, tags, labels)

if __name__ == '__main__':
    tags, tag_string = get_tags()
    conf = get_conf(tags)
    conf.tag_string = tag_string

    init_logging()
    log = get_logger()

    #lambda_vals = [0.01, 1, 100]
    #t0_vals = [1, 10, 100, 1000]
    #n_vals = [1, 5, 10, 25]
    p_vals = [2]
    splits = 5
    results = np.empty([len(p_vals),splits])
    savefile=tags.get('tag', "results")
    for ni, p_val in enumerate(p_vals):
        for s in range(splits):
            tags['p'] = p_val
            get_logger().info("Testing P " + str(tags.get('p', 10)) + " split " + str(s))
            params = dict(lambda_ = float(tags.get('lambda', 1.0)), #1/100 1 100
                          n       = int(tags.get('n', 10)),
                          t0      = float(tags.get('t0', 1)), #1 10- 100 - 1000
                          p       = float(tags.get('p', 2.0)),
                          L       = float(tags.get('L', 1.0)),
                          model_path = conf.model_path,
                          save_model_every_t = int(tags.get('save_every', 500)),
                          model_tag = tags['tag'],
                          nbnl_mode = True)


            cls = SmoothML3(**params)

            if tags['do'] == 'Train':
                data_indexes = get_indexes(tags.get('input_folder'), int(tags.get('train_images',50)), int(tags.get('test_images',50)) )
                num_classes = len(data_indexes.train)
                training_data = load_dataset(data_indexes.train)
                train_loader = MemoryLoader(training_data.patches, training_data.tags, training_data.labels, \
                                             int(tags.get('passes', 3)), int(tags.get('batch', 1024)), trainMode = True)
                del training_data
                cls.fit_from_loader(loader=train_loader,
                                    num_classes=num_classes)
                train_scaler = train_loader.scaler
                del train_loader
                test_data = load_dataset(data_indexes.test)
                test_loader = MemoryLoader(test_data.patches, test_data.tags, test_data.labels, \
                                             0, 32768, trainMode = False, scaler=train_scaler)
                del test_data
                score = cls.score_from_loader(test_loader,
                                              on_gpu=True,
                                              nbnl=True)
                test_loader.reset()
                #patch_score = cls.score_from_loader(test_loader,
                                              #on_gpu=True,
                                              #nbnl=False)
                get_logger().info("Score " + str(score) + " (" + str(0) + ")")
                results[ni,s] = score
            elif tags['do'] == 'resume':
                model_files = filter(lambda s: os.path.isfile(s) and tags['tag'] in s, glob(conf.model_path + '/*'))
                if len(model_files) == 0:
                    log.error('No model files found to resume.')
                    exit()

                model_files.sort(key=lambda x: os.path.getmtime(x))
                last_created_model_filename = model_files[-1]
                last_batch_i = int(re.findall('t=(\\d+)', last_created_model_filename)[0])

                train_files = sorted(glob(conf.train_target + '/*.npz'))
                loader_params = dict(files  = train_files,
                                     passes = int(tags.get('passes', 3)),
                                     minibatch_size = int(tags.get('batch', 1024)),
                                     shuffle_examples = True,
                                     use_positions = 'dont-use-positions' not in tags,
                                     nmz = tags.get('nmz', 'std'),
                                     reg_bias = int(tags.get('reg-bias', '0')),
                                     label_correction=-1)
                loader = ILSVRCPatchMinibatchLoader(**loader_params)
                loader.init_scaler()
                log.info('Looking for minibatch to resume...')
                loader.rewind_to_minibatch(last_batch_i)
                log.info('Found.')
                cls.resume_from_loader(loader=loader,
                                       model_filename=last_created_model_filename)

            elif tags['do'] == 'test':
                train_files = sorted(glob(conf.train_target + '/*.npz'))
                test_files = sorted(glob(conf.val_target + '/*.npz'))
                loader_params = dict(files  = test_files,
                                     passes = int(tags.get('passes', 1)),
                                     minibatch_size = int(tags.get('batch', 1024)),
                                     shuffle_examples = False,
                                     use_positions = 'dont-use-positions' not in tags,
                                     nmz = tags.get('nmz', 'std'),
                                     reg_bias = int(tags.get('reg-bias', '0')),
                                     label_correction=-1)
                loader = ILSVRCPatchMinibatchLoader(**loader_params)
                loader.init_scaler(train_files[0])

                tag_pattern = []
                if 'tag' in tags:
                    tag_pattern.append('%(tag)s')
                if 'p' in tags:
                    tag_pattern.append('p=%(p)s')
                if 'n' in tags:
                    tag_pattern.append('n=%(n)s')
                if 'lambda' in tags:
                    tag_pattern.append('lambda=%(lambda)s')
                if 't0' in tags:
                    tag_pattern.append('t0=%(t0)s')
                if 'L' in tags:
                    tag_pattern.append('L=%(L)s')


                model_files = filter(lambda s: os.path.isfile(s), glob(conf.model_path + '/*' + '*'.join(tag_pattern) % tags + '*'))
                if len(model_files) == 0:
                    log.error('No model files found to load from.')
                    exit()

                model_files.sort(key=lambda x: os.path.getmtime(x))
                last_created_model_filename = model_files[-1]

                use_gpu = 'cpu' not in tags

                cls.load(last_created_model_filename,
                         to_gpu=use_gpu)


                score = cls.score_from_loader(loader=loader,
                                              on_gpu=use_gpu,
                                              nbnl=True)

                print str( dict(cls_params=params, loader_params=loader_params, score=score) )
    with open(savefile + ".pickle", 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

