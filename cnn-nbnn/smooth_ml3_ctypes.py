import copy
import numpy as np
import numpy.ctypeslib as npct
from scipy.sparse import csr_matrix, issparse
from ctypes import c_uint, c_int, c_uint32, c_uint64, c_float, c_double, c_char_p, c_void_p, CFUNCTYPE, c_bool, Structure, POINTER

array1f = npct.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
array2f = npct.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
array1u = npct.ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS')

class SparseDataBatch(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("col_index", POINTER(c_int)),
        ("row_ptr", POINTER(c_int)),
        ("length", c_int),
        ("cols", c_int),
        ("rows", c_int),
        ("y", POINTER(c_uint32)),
        ("finished", c_bool),
        ]

class DenseDataBatch(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("cols", c_int),
        ("rows", c_int),
        ("y", POINTER(c_uint32)),
        ("tags", POINTER(c_uint32)),
        ("finished", c_bool),
        ]

sparse_loader_func = CFUNCTYPE( None, POINTER(SparseDataBatch) )
dense_loader_func = CFUNCTYPE( None, POINTER(DenseDataBatch) )

cu_libml3 = npct.load_library("lib_cu_smooth_ml3_ctypes", "smooth_ml3_ctypes")

cu_libml3.train_smooth_ml3_ctypes.restype = c_void_p
cu_libml3.train_smooth_ml3_ctypes.argtypes = [dense_loader_func, c_uint32, c_uint32, c_float, c_float, c_float, c_float, c_float, c_uint64, c_bool, c_char_p, c_uint32, c_char_p, dense_loader_func, c_bool]

cu_libml3.resume_smooth_ml3_ctypes.restype = c_void_p
cu_libml3.resume_smooth_ml3_ctypes.argtypes = [dense_loader_func, c_char_p, c_char_p, c_uint32, c_char_p, dense_loader_func, c_bool]

cu_libml3.test_smooth_ml3_ctypes.restype = c_float
cu_libml3.test_smooth_ml3_ctypes.argtypes = [c_void_p, dense_loader_func, c_bool, c_bool]

cu_libml3.destroy_smooth_ml3_ctypes.restype = None
cu_libml3.destroy_smooth_ml3_ctypes.argtypes = [c_void_p]

cu_libml3.get_train_time_ms_ctypes.restype = c_double
cu_libml3.get_train_time_ms_ctypes.argtypes = [c_void_p]

cu_libml3.get_test_time_ms_ctypes.restype = c_double
cu_libml3.get_test_time_ms_ctypes.argtypes = [c_void_p]

cu_libml3.get_aer_track_length.restype = c_uint64
cu_libml3.get_aer_track_length.argtypes = [c_void_p]

cu_libml3.get_aer_track.restype = None
cu_libml3.get_aer_track.argtypes = [c_void_p, POINTER(c_float)]

cu_libml3.load_model_ctypes.restype = c_void_p
cu_libml3.load_model_ctypes.argtypes = [c_char_p]

class SparseDataLoader(object):

    def __init__(self, loader_fn):
        self.loader_fn = loader_fn

    def loader(self, batch):
        X, y = self.loader_fn()

        if X is None:
            batch.contents.finished = c_bool(True)
        else:
            assert(isinstance(X, csr_matrix))

            batch.contents.finished = c_bool(False)

            self.data = np.ascontiguousarray(X.data.astype(np.float32))
            self.col_index = np.ascontiguousarray(X.indices.astype(np.int32))
            self.row_ptr = np.ascontiguousarray(X.indptr.astype(np.int32))
            self.y = np.ascontiguousarray(y.astype(np.int32))

            if self.y is not None:
                self.y = np.ascontiguousarray(self.y.astype(np.uint32))

            batch.contents.data = self.data.ctypes.data_as(POINTER(c_float))
            batch.contents.col_index = self.col_index.ctypes.data_as(POINTER(c_int))
            batch.contents.row_ptr = self.row_ptr.ctypes.data_as(POINTER(c_int))
            batch.contents.rows = c_int(X.shape[0])
            batch.contents.cols = c_int(X.shape[1])
            batch.contents.length = self.data.size
            batch.contents.y = self.y.ctypes.data_as(POINTER(c_uint32))

class DenseDataLoader(object):

    def __init__(self, loader_obj, reset_when_finished=False):
        self.loader_obj = loader_obj
        self.reset_when_finished = reset_when_finished
        if reset_when_finished:
            self.loader_obj_copy = copy.copy(self.loader_obj)

    def loader(self, batch):
        X, y, tags = self.loader_obj.loader()

        if X is None:
            batch.contents.finished = c_bool(True)

            if self.reset_when_finished:
                print 'Loader finished. Resetting to start.'
                self.loader_obj = copy.copy(self.loader_obj_copy)
        else:
            assert(not issparse(X))

            batch.contents.finished = c_bool(False)

            self.data = X.astype(np.float32, order='C')
            self.y = np.ascontiguousarray(y.astype(np.uint32))

            if tags is not None and len(tags) > 0:
                self.tags = np.ascontiguousarray(tags.astype(np.uint32))
            else:
                self.tags = None

            if self.y is not None:
                self.y = np.ascontiguousarray(self.y.astype(np.uint32))

            batch.contents.data = self.data.ctypes.data_as(POINTER(c_float))
            batch.contents.rows = c_int(X.shape[0])
            batch.contents.cols = c_int(X.shape[1])
            batch.contents.y = self.y.ctypes.data_as(POINTER(c_uint32))

            if self.tags is not None and len(self.tags) > 0:
                batch.contents.tags = self.tags.ctypes.data_as(POINTER(c_uint32))
            else:
                batch.contents.tags = None


def train_cu_smooth_ml3(loader, classes, n, L, t0, p, lambda_, gauss_init_std=0.01, pass_length=0, record_aer=False,
                        model_path=0, save_model_every_t=0, model_tag=0, validation_loader=None, nbnl=False):
    dense_loader = DenseDataLoader(loader)
    dense_validation_loader_p = dense_loader_func(0)
    if validation_loader:
        dense_validation_loader = DenseDataLoader(validation_loader, reset_when_finished=True)
        dense_validation_loader_p = dense_loader_func(dense_validation_loader.loader)

    ml3_hand = cu_libml3.train_smooth_ml3_ctypes(dense_loader_func(dense_loader.loader),
                                                 classes, n,
                                                 L, t0, p, lambda_, gauss_init_std, pass_length, record_aer,
                                                 c_char_p(model_path), save_model_every_t, c_char_p(model_tag), dense_validation_loader_p, nbnl)
    return ml3_hand

def resume_cu_smooth_ml3(loader, model_filename, model_path=0, save_model_every_t=0, model_tag=0, validation_loader=None, nbnl=False):
    dense_loader = DenseDataLoader(loader)
    dense_validation_loader_p = dense_loader_func(0)
    if validation_loader:
        dense_validation_loader = DenseDataLoader(validation_loader)
        dense_validation_loader_p = dense_loader_func(dense_validation_loader.loader)

    ml3_hand = cu_libml3.resume_smooth_ml3_ctypes(dense_loader_func(dense_loader.loader),
                                                  c_char_p(model_filename),
                                                  c_char_p(model_path), save_model_every_t, c_char_p(model_tag), dense_validation_loader_p, nbnl)
    return ml3_hand
    
def test_cu_smooth_ml3(ml3_hand, loader, on_gpu=True, nbnl=False):
    dense_loader = DenseDataLoader(loader)
    return cu_libml3.test_smooth_ml3_ctypes(ml3_hand, dense_loader_func(dense_loader.loader), on_gpu, nbnl)

def destroy_cu_smooth_ml3(ml3_hand):
    cu_libml3.destroy_smooth_ml3_ctypes(ml3_hand)


def cu_load_model(model_filename, to_gpu=True):
    return cu_libml3.load_model_ctypes(c_char_p(model_filename), c_bool(to_gpu))

def get_cu_train_time_ms(ml3_hand):
    return cu_libml3.get_train_time_ms_ctypes(ml3_hand)

def get_cu_test_time_ms(ml3_hand):
    return cu_libml3.get_test_time_ms_ctypes(ml3_hand)

def get_cu_aer_track_length(ml3_hand):
    return cu_libml3.get_aer_track_length(ml3_hand)

def get_cu_aer_track(ml3_hand):
    aer_track = np.zeros((get_cu_aer_track_length(ml3_hand),), dtype=np.float32)

    cu_libml3.get_aer_track(ml3_hand, aer_track.ctypes.data_as(POINTER(c_float)))
    return aer_track
