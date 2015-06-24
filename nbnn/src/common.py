
import sys
import logging
import os
from os.path import join
import uuid
from os import popen
from ConfigParser import ConfigParser

def init_logging():
    logging.StreamHandler(sys.stdout)
    log = get_logger()
    log.setLevel(level=logging.DEBUG)

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(funcName)s: \"%(message)s\"",
                                  "%Y-%m-%d %H:%M:%S")
    stream.setFormatter(formatter)
    log.addHandler(stream)

def get_logger():
    return logging.getLogger('main')

def get_root_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

def launch_on_the_grid(sge_conf, job_conf):
    """ Launches the job on the grid according to the job_conf spec.
    Input:
    job_conf -- a dictionary with the following fields,
        shell -- command to execute at each grid node
        grid_args -- keys of job_conf entries to be partitioned on the grid.
                     Each such entry is a list. Taking cartesian product of all these lists, we assign
                     each element of the product to the grid node as a configuration.

    sge_conf -- grid job configuration, a dictionary with the following fields
        job_dir -- a directory to store temporary job scripts
        log_dir -- a directory to output working logs
        err_log_dir -- a directory to output error logs
        sge_spec -- additional parameters to be passed to qsub, queue, memory, etc.
        depends_on -- name of the job, this job depends on. """

    job_conf_path = join(sge_conf['job_dir'], 'job_conf_%s.txt' % str(uuid.uuid4()))
    job_conf_file = open(job_conf_path, 'wt')
    job_conf_file.write(str(job_conf))
    job_conf_file.close()

    conf = sge_conf.copy()
    conf['num_tasks'] = reduce(lambda x,y: x*y,
                               map(lambda k: len(job_conf[k]), job_conf['grid_args']))

    conf['job']= join(conf['job_dir'], 'tmp_launch_job.sh')
    open(conf['job'], 'wt').write('%s %s' % (join(get_root_path(), 'src/bootstrap_job.py'), job_conf_path))

    if 'depends_on' in sge_conf:
        conf['sge_spec'] += ' -hold_jid %s' % sge_conf['depends_on']

    sge_shell = '''
    export PATH=$PATH:%(sge_path)s
    qsub -N %(name)s -e %(err_log_dir)s -o %(log_dir)s %(sge_spec)s -t 1-%(num_tasks)s "%(job)s"
''' % conf

    tmp_launch = join(conf['job_dir'], 'tmp_launch.sh')
    tmp_f = open(tmp_launch, 'wt')
    tmp_f.write(sge_shell)
    tmp_f.close()

    log = get_logger()
    if not sge_conf.get('silent', False):
        log.info('Please inspect following script and provide decision:');
        log.info('-----------------------------------------------------------------')
        log.info('Grid launch script:')
        log.info('-----------------------------------------------------------------')
        log.info(sge_shell)
        log.info('-----------------------------------------------------------------')

        log.info('PROVIDE DECISION [Y/N]: ')

        inp = raw_input()
        if inp.lower() == 'y':
            os.popen('bash "%s"' % tmp_launch)
    else:
        os.popen('bash "%s"' % tmp_launch)
        log.info('Launched job <%s>.' % sge_conf['name'])

def get_config(config_file, key):
    cp = ConfigParser()
    cp.read(config_file)
    return dict(cp.items(key)) if cp.has_section(key) else dict()

def kill_jobs(str_range):
    if ':' in str_range:
        ids = str_range.split(':')
        popen('qdel ' + ' '.join(map(str, range(int(ids[0]), int(ids[1])+1))))

def get_desc_name(conf):
    if conf['descriptor'] == 'SIFT':
        desc_type = 'DSIFT'
    elif conf['descriptor'] == 'DECAF':
        desc_type = 'DECAF'
    else:
        raise Error, 'Unsupported descriptor type.'
    conf_ = conf.copy()
    conf_['desc_type'] = desc_type
    conf_['oversample'] = bool(int(conf_['oversample']))
    conf_['decaf_oversample'] = bool(int(conf_['decaf_oversample']))

    conf_['layer_tag'] = { 'fc6_cudanet_out': '6',
                           'fc6_neuron_cudanet_out': '6relu',
                           'fc7_cudanet_out': '7',
                           'fc7_neuron_cudanet_out': '7relu',
                           }.get(conf_['decaf_layer'], conf_['decaf_layer'])

    return '%(desc_type)s_TRAIN_%(num_train_images)s_TEST_%(num_test_images)s_PATCHES_%(patches_per_image)s_SIZE_%(patch_size)s_LEVELS_%(levels)s_IM_DIM_%(image_dim)s_OVERSAMPLE_%(oversample)s_DECAF_OVERSAMPLE_%(decaf_oversample)s_LAYER_%(layer_tag)s' % conf_

def get_result_name(conf, support_type):
    conf_ = conf.copy()

    common_token = 'TRAIN_%(num_train_images)s_TEST_%(num_test_images)s_ALPHA_%(alpha)s' % conf_
    conf_['common_token'] = common_token

    if conf_['alg'] == 'nn' and int(conf_['knn']) > 1:
        conf_['alg_type'] = '%(knn)s_%(alg_type)s' % conf_

    if support_type == 'select':
        conf_['stabilized'] = '_STABILIZED' if int(conf_.get('stabilized', 0)) > 0 else ''
        conf_['all_cand'] = '' if int(conf_.get('59_trick', 1)) else '_ALL_CAND'
        return 'SUPPORT%(stabilized)s%(all_cand)s_%(alg)s_%(alg_type)s_%(common_token)s_SUPPORT_SIZE_%(support_size)s_GAMMA_%(gamma)s' % conf_
    elif support_type == 'random':
        return 'RANDOM_SUPPORT_%(alg)s_%(alg_type)s_%(common_token)s_SUPPORT_SIZE_%(support_size)s' % conf_
    elif support_type == 'full':
        return 'ALL_%(alg)s_%(alg_type)s_%(common_token)s' % conf_
