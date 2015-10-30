
import sys
import logging
import os
from os.path import join, exists, basename, splitext
from glob import glob
import uuid
from os import popen
from ConfigParser import ConfigParser

import numpy as np
from scipy.io import loadmat, savemat

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

def get_cur_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ )))

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
    open(conf['job'], 'wt').write('%s %s' % (join(get_cur_path(), 'bootstrap_job.py'), job_conf_path))

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
            return True
        else:
            return False
    else:
        os.popen('bash "%s"' % tmp_launch)
        log.info('Launched job <%s>.' % sge_conf['name'])
        return True

def get_sge_status(sge_config):
    sge_shell = '''
    export PATH=$PATH:%(sge_path)s
    qstat
''' % sge_config
    os.popen(sge_status)

def get_config(config_file, key):
    cp = ConfigParser()
    cp.read(config_file)
    return dict(cp.items(key)) if cp.has_section(key) else dict()

def kill_jobs(str_range):
    if ':' in str_range:
        ids = str_range.split(':')
        popen('qdel ' + ' '.join(map(str, range(int(ids[0]), int(ids[1])+1))))

def translate_labels(y):
    # Translating labels to 0, 1, 2, ...
    classes = set(y)
    class_map = dict(zip(sorted(classes), range(len(classes))))
    return (map(class_map.get, y), set(y))

def is_power2(num):
    'states if a number is a power of two'

    return num != 0 and ((num & (num - 1)) == 0)

def add_reg_bias(X_):
    m, d = X_.shape
    return np.hstack([X_, -np.ones( (m, 1) )])

def print_benchmark_results_compact(rs, show_class_ratios=False):
    def convert(x):
        if isinstance(x,bool):
            return int(x)
        elif isinstance(x,str):
            return '"%s"' % x
        elif isinstance(x,np.ndarray) or isinstance(x,np.matrix):
            return x.tolist()
        else:
            return x

    print '; '.join(['%s:%s' % (k,convert(v)) for k,v in rs.items()])
    sys.stdout.flush()

def get_tags(tag_string_delim='_', ignore_in_tag_string=[]):
    tags = sys.argv[1].split(',') if len(sys.argv) > 1 else []
    def parse_tag(s):
        if '=' in s:
            return tuple(s.split('='))
        else:
            return (s,None)

    tags_ = dict(map(parse_tag, tags))
    tags = list(set(tags).difference(set(ignore_in_tag_string)))
    tag_string = tag_string_delim.join(sorted(tags))
    return (tags_, tag_string)

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def get_conf(tags):
    log = get_logger()
    conf = eval(open('conf.py').read())
    _conf = conf['params']

    operative_key = set(tags).intersection(set(conf.keys()))
    if len(operative_key) == 1:
        operative_key = list(operative_key)[0]
        log.info('Running %s...', operative_key)
        _conf.update(conf[operative_key])
        _conf['operative_key'] = operative_key
    else:
        log.info('Could not identify one operative key in tags.')

    _conf['tags'] = tags

    return Bunch(_conf)

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
