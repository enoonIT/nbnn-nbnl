# -*- coding: utf-8 -*-
from h5py import File as HDF5File
from common import get_logger
import numpy as np

class ClassPatches:
    def __init__(self, filename, indexes, patch_name):
        self.file_name = filename
        self.indexes = indexes
        self.patches = None
        self.patch_name = patch_name
    def get_patches(self):
        if self.patches is None:
            self.load()
        return self.patches
    def load(self):
        get_logger().info("Loading patches for " + self.file_name)
        hfile = HDF5File(self.file_name, 'r')
        patches = hfile[self.patch_name]
        feature_dim = patches.shape[1]
        indexes = self.indexes
        num_patches=(indexes[:,1]-indexes[:,0]).sum()
        self.patches = np.empty([num_patches, feature_dim])
        patch_start = 0
        for iid in indexes:
            n_patches = iid[1]-iid[0]
            self.patches[patch_start:patch_start+n_patches,:] = patches[iid[0]:iid[1],:]
            patch_start += n_patches
        hfile.close()
        get_logger().info("Loaded " + str(num_patches) + " patches")
    def unload(self):
        get_logger().info("Unloading patches for " + self.file_name)
        self.patches = None
    def get_num_patches(self):
        return (self.indexes[:,1]-self.indexes[:,0]).sum()