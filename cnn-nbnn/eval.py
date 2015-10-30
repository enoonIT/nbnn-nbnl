
from SmoothML3 import SmoothML3
import sklearn
import random
import copy
import os
import re

from common import *
from data import ILSVRCPatchMinibatchLoader

if __name__ == '__main__':
    tags, tag_string = get_tags()
    conf = get_conf(tags)
    conf.tag_string = tag_string

    init_logging()
    log = get_logger()

    params = dict(lambda_ = float(tags.get('lambda', 1.0)),
                  n       = int(tags.get('n', 10)),
                  t0      = float(tags.get('t0', 1)),
                  p       = float(tags.get('p', 2.0)),
                  L       = float(tags.get('L', 1.0)),
                  model_path = conf.model_path,
                  save_model_every_t = int(tags.get('save_every', 500)),
                  model_tag = tags['tag'],
                  nbnl_mode = True)


    cls = SmoothML3(**params)

    if tags['do'] == 'train':
        train_files = sorted(glob(conf.train_target + '/*.npz'))
        loader_params = dict(files  = train_files,
                             passes = int(tags.get('passes', 3)),
                             minibatch_size = int(tags.get('batch', 1024)),
                             shuffle_examples = True,
                             use_positions = 'dont-use-positions' not in tags,
                             use_pct_of_files = int(tags.get('file-pct', 100)),
                             nmz = tags.get('nmz', 'std'),
                             reg_bias = int(tags.get('reg-bias', '0')),
                             label_correction=-1)
        loader = ILSVRCPatchMinibatchLoader(**loader_params)
        loader.init_scaler()
        cls.fit_from_loader(loader=loader,
                            num_classes=1000)

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
        
