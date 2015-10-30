import sys
import numpy as np
from random import shuffle
import random
from scipy.sparse import csr_matrix, issparse
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from smooth_ml3_ctypes import POINTER, c_float, c_uint32, c_bool, \
    get_cu_train_time_ms, get_cu_test_time_ms, get_cu_aer_track, \
    train_cu_smooth_ml3, resume_cu_smooth_ml3, test_cu_smooth_ml3, destroy_cu_smooth_ml3, cu_load_model

from common import *

class SparseOneShotLoader(object):
   def __init__(self, passes, _X, _y):
       self.passes = passes
       self._X = _X
       self._y = _y
   def loader(self):
       if self.passes > 0:
           shuffle_ix = range(len(self._y))
           shuffle(shuffle_ix)
           sparse_mat = csr_matrix(self._X[shuffle_ix, :])
           self.passes -= 1
           return sparse_mat, self._y[shuffle_ix]-1
       else:
           return None, None

class DenseOneShotLoader(object):
   def __init__(self, passes, _X, _y, _tags, transpose=False):
       self.passes = passes
       self._X = _X
       self._y = _y
       self._tags = _tags
       self.transpose = transpose
   def loader(self):
       if self.passes > 0:
           shuffle_ix = range(len(self._y))
           shuffle(shuffle_ix)
           mat = self._X[shuffle_ix, :]
           self.passes -= 1
           return mat.T, self._y[shuffle_ix]-1, self._tags[shuffle_ix]
       else:
           return None, None, None

class MinibatchDenseOneShotLoader(object):
   def __init__(self, passes, _X, _y, _tags, transpose=False, minibatch_size=2500, shuffle=True):
       self.passes = passes
       self._X = _X
       self._y = _y-1
       self._tags = _tags
       self.ix = np.arange(len(self._y))

       self.transpose = transpose
       self.minibatch_size = minibatch_size

       self.cursor = 0
       self.do_shuffle = shuffle

       if self.do_shuffle:
          self.shuffle()

   def loader(self):
       if self.passes > 0:
           start = self.cursor
           end = min(self.cursor + self.minibatch_size, self._X.shape[0])

           mat = self._X[self.ix[start:end]]
           labels = self._y[self.ix[start:end]]

           if self._tags is not None and len(self._tags) > 0:
              tags = self._tags[self.ix[start:end]]
           else:
              tags = None

           if self.transpose:
               mat = mat.T

           if end == self._X.shape[0]:
               self.passes -= 1
               self.cursor = 0
               if self.do_shuffle:
                  self.shuffle()
           else:
               self.cursor += self.minibatch_size

           return mat, labels, tags
       else:
           return None, None, None

   def shuffle(self):
       shuffle(self.ix)

class SmoothML3(BaseEstimator, ClassifierMixin):

   def __init__(self, passes=10, lambda_=0.0, n=10, t0=0.0, p=2.0, L=1.0, gauss_init_std=1, minibatch_size=2500, pass_length=0, record_aer=True, ml3_hand=None,
                 model_path=None, save_model_every_t=0, model_tag=None, nbnl_mode=False):
      self.passes = passes
      self.lambda_ = lambda_
      self.gauss_init_std = gauss_init_std
      self.t0 = t0
      self.n = n
      self.p = p
      self.L = L
      self.minibatch_size = minibatch_size
      self.record_aer = record_aer
      self.ml3_hand = ml3_hand
      self.pass_length = pass_length
      self.model_path=model_path
      self.save_model_every_t=save_model_every_t
      self.model_tag = model_tag
      self.nbnl_mode = nbnl_mode

   def get_params(self, deep=True):
      return {'passes': self.passes, 'lambda_': self.lambda_, 't0': self.t0, 'n': self.n, 'p': self.p, 'L': self.L, 'minibatch_size': self.minibatch_size,
              'ml3_hand': self.ml3_hand, 'record_aer': self.record_aer, 'pass_length': self.pass_length,
              'model_path': self.model_path, 'model_tag': self.model_tag, 'save_model_every_t': self.save_model_every_t, 'nbnl_mode': self.nbnl_mode}

   def set_params(self, **parameters):
      for parameter, value in parameters.items():
         setattr(self, parameter, value)
      return self

   def fit(self, X, y):
      X = add_reg_bias(X)

      log = get_logger()

      loader = MinibatchDenseOneShotLoader(self.passes, X, y, transpose=True, minibatch_size=self.minibatch_size)
      train = train_cu_smooth_ml3

      destroy_cu_smooth_ml3(self.ml3_hand)
      self.ml3_hand = train(loader.loader,
                            classes=len(np.unique(y)),
                            n=self.n,
                            L=self.L,
                            t0=self.t0,
                            p=self.p,
                            lambda_=self.lambda_,
                            gauss_init_std=self.gauss_init_std,
                            pass_length=self.pass_length,
                            record_aer=self.record_aer,
                            model_path=self.model_path,
                            save_model_every_t=self.save_model_every_t,
                            model_tag=self.model_tag,
                            validation_loader=None,
                            nbnl=self.nbnl_mode)
      self.train_time_ms = get_cu_train_time_ms(self.ml3_hand)

   def fit_from_loader(self, loader, num_classes, validation_loader=None):
      destroy_cu_smooth_ml3(self.ml3_hand)

      self.ml3_hand = train_cu_smooth_ml3(loader,
                                          classes=num_classes,
                                          n=self.n,
                                          L=self.L ,
                                          t0=self.t0,
                                          p=self.p,
                                          lambda_=self.lambda_,
                                          gauss_init_std=self.gauss_init_std,
                                          pass_length=self.pass_length,
                                          record_aer=self.record_aer,
                                          model_path=self.model_path,
                                          save_model_every_t=self.save_model_every_t,
                                          model_tag=self.model_tag,
                                          validation_loader=validation_loader,
                                          nbnl=self.nbnl_mode)
      self.train_time_ms = get_cu_train_time_ms(self.ml3_hand)


   def resume_from_loader(self, loader, model_filename):
      destroy_cu_smooth_ml3(self.ml3_hand)

      self.ml3_hand = resume_cu_smooth_ml3(loader,
                                           model_filename=model_filename,
                                           model_path=self.model_path,
                                           save_model_every_t=self.save_model_every_t,
                                           model_tag=self.model_tag)
      self.train_time_ms = get_cu_train_time_ms(self.ml3_hand)


   def score_from_loader(self, loader, on_gpu=True, nbnl=False):
      score = test_cu_smooth_ml3(self.ml3_hand, loader, on_gpu, nbnl)
      self.test_time_ms = get_cu_test_time_ms(self.ml3_hand)
      destroy_cu_smooth_ml3(self.ml3_hand)
      return score

   def score(self, X, y, on_gpu=True):
      X = add_reg_bias(X)

      loader = MinibatchDenseOneShotLoader(1, X, y, transpose=True, minibatch_size=self.minibatch_size)
      test = test_cu_smooth_ml3

      score = test(self.ml3_hand, loader.loader, on_gpu)
      self.test_time_ms = get_cu_test_time_ms(self.ml3_hand)

      destroy_cu_smooth_ml3(self.ml3_hand)

      return score

   def finish(self):
      destroy_cu_smooth_ml3(self.ml3_hand)

   def get_aer_track(self):
      return get_cu_aer_track(self.ml3_hand)

   def load(self, filename, to_gpu=True):
      self.ml3_hand = cu_load_model(filename, to_gpu)
      self.train_time_ms = get_cu_train_time_ms(self.ml3_hand)
