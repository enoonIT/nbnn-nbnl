import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from common import *
import random

class MemoryLoader(object):
    def __init__(self, patches, tags, labels, passes, minibatch_size=1024, scaler=None, trainMode = True):
        self.reset()
        self.max_passes = passes
        self.minibatch_size = minibatch_size
        self.tags = tags
        #import pdb; pdb.set_trace()
        self.patches = patches
        del patches
        self.labels = labels
        self.ordering = np.arange(self.patches.shape[0])
        self.trainMode = trainMode
        get_logger().info(self.patches.shape)
        get_logger().info("std " + str(self.patches.std(0).shape) + " / " + str(self.patches.std(0).mean()) + \
                    " mean " + str(self.patches.mean(0).shape) + " / " + str(self.patches.mean(0).mean()))
        if trainMode:
            self.scaler = StandardScaler().fit(self.patches)
            get_logger().info("Creating scaler")
            np.random.shuffle(self.ordering)
        else: #test mode
            self.scaler = scaler
            self.image_location = self.tags.nonzero()[0]
        if(self.scaler is not None):
            self.patches = self.scaler.transform(self.patches)
            get_logger().info("Applying scaler")
        get_logger().info("std " + str(self.patches.std(0).mean()) + " mean " + str(self.patches.mean(0).mean()))
    def reset(self):
        self.cursor = 0
        self.current_pass = 0
    def loader(self): #patches x dimensions
        #get_logger().info("Loading new batch")
        remaining = self.ordering.shape[0] - self.cursor
        if(remaining <= 0):
            if(self.current_pass >= self.max_passes):
                return (None, None, None)
            else:
                get_logger().info("Pass " + str(self.current_pass) + " for loader")
                self.current_pass += 1
                self.cursor = 0
                if self.trainMode: np.random.shuffle(self.ordering)
                remaining = self.ordering.shape[0]
        toGet = min(remaining, self.minibatch_size)
        if not self.trainMode: #testing mode
            #import pdb; pdb.set_trace()
            get_logger().info("Should get " + str(toGet) + " finishing at " + str(self.cursor + toGet))
            if toGet != remaining:
                lastPatch = self.cursor + toGet
                nextImage = np.where(self.image_location >= lastPatch)[0]
                if nextImage.size: #if not the last image
                    endImage = self.image_location[nextImage[0]]
                    toGet = endImage - self.cursor
                else: #get all the remaining patches
                    toGet = remaining
            get_logger().info("Will get " + str(toGet) + " finishing at " + str(self.cursor + toGet))
        indexes = self.ordering[self.cursor:self.cursor+toGet]
        self.cursor += toGet
        X = self.patches[indexes,:]
        y = self.labels[indexes].astype(dtype=np.uint32).ravel()
        tags = self.tags[indexes].astype(dtype=np.uint32)
        if not self.trainMode:
            assert tags[0]!=0, "First tag element should always be 1 during testing"
        return (X.T, y, tags)


class Cifar10MinibatchLoader(object):
    '''
    This class implements reading of batch files and on-the-fly splitting into minibatches.
    The method loader() is supposed to be called repeatedly by minibatch consumer until None, None is returned.
    init_scaler() must be called before first call to loader(), that is to initialize normalization parameters.
    '''

    def __init__(self, files, passes, minibatch_size=1024, scaler=None):
        self.filenames = sorted(files)
        self.n_files = len(self.filenames)
        self.i = 0

        self.passes = passes
        self.minibatch_size = minibatch_size

        self.X = None
        self.y = None
        self.current_file_size = 0
        self.cursor = 0
        self.scaler = scaler

    def init_scaler(self, filename=None):
        if filename is None:
            mat = loadmat(self.filenames[0])
        else:
            mat = loadmat(filename)

        X = mat['data'].astype(np.float32)
        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(X)

    def loader(self):
        if self.cursor >= self.current_file_size: # Depleted current file, will read a new one.
            if self.i == self.passes*self.n_files: # Or, if no files left to feed just pass None, None.
                return None, None, None

            if 0 == (self.i % self.n_files):
                get_logger().info('Pass: %s', (self.i / self.n_files)+1)

            mat = loadmat(self.filenames[self.i % self.n_files])

            self.X = mat['data']
            self.y = mat['labels'].ravel().astype(dtype=np.uint32)
            self.tags = np.zeros((len(self.y),), dtype=np.uint32)

            self.current_file_size = self.X.shape[0]
            self.cursor = 0
            self.i += 1


        # Getting next minibatch from current file
        start = self.cursor
        end = min(self.cursor+self.minibatch_size, self.current_file_size)
        X = self.X[start:end, :]
        y = self.y[start:end]
        tags = self.tags[start:end]
        self.cursor += self.minibatch_size

        # Normalizing the data
        X = X.astype(np.float32)
        self.scaler.transform(X)
        #X = add_reg_bias(X) # Regularized bias term

        return X.T, y, tags


class ILSVRCMinibatchLoader(object):


    def __init__(self, files, passes, minibatch_size=1024, scaler=None,
                 shuffle_examples=True, use_pct_of_files=100, nmz='std', reg_bias=0):
        self.filenames = sorted(files)
        self.n_files = len(self.filenames)
        self.i = 0

        self.passes = passes
        self.minibatch_size = minibatch_size

        self.X = None
        self.y = None
        self.current_file_size = 0
        self.cursor = 0
        self.scaler = scaler
        self.shuffle_examples = shuffle_examples
        self.nmz = nmz
        self.reg_bias = reg_bias

    def init_scaler(self, filename=None):
        if filename is None:
            mat = np.load(self.filenames[0])
        else:
            mat = np.load(filename)
        X = mat['patches'].item().todense().astype(np.float32)
        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(X)

    def loader(self):
        if self.cursor >= self.current_file_size: # Depleted current file, will read a new one.
            if self.i == self.passes*self.n_files: # Or, if no files left to feed just pass None, None.
                return None, None, None

            if 0 == (self.i % self.n_files):
                get_logger().info('Pass: %s', (self.i / self.n_files)+1)

            mat = np.load(self.filenames[self.i % self.n_files])

            self.X = mat['patches'].item()
            self.y = mat['labels'].ravel().astype(dtype=np.uint32)
            self.tags = np.zeros((len(self.y),), dtype=np.uint32)

            self.current_file_size = self.X.shape[0]
            self.cursor = 0
            self.i += 1


        # Getting next minibatch from current file
        start = self.cursor
        end = min(self.cursor+self.minibatch_size, self.current_file_size)
        X = self.X[start:end, :].todense().astype(np.float32)
        y = self.y[start:end]
        tags = self.tags[start:end]
        self.cursor += self.minibatch_size

        # Normalizing the data
        X = X.astype(np.float32)

        if self.nmz != 'no':
            self.scaler.transform(X)

        if self.reg_bias:
            X = add_reg_bias(X) # Regularized bias term

        return X.T, y, tags

class ILSVRCPatchMinibatchLoader(object):
    '''
    This class implements reading of batch files and on-the-fly splitting into minibatches.
    The method loader() is supposed to be called repeatedly by minibatch consumer until None, None is returned.
    init_scaler() must be called before first call to loader(), that is to initialize normalization parameters.
    '''

    def __init__(self, files, passes, minibatch_size=1024, scaler=None, shuffle_examples=True, use_positions=True, use_pct_of_files=100, nmz='std', reg_bias=0, label_correction=-1, use_levels=[]):
        self.filenames = sorted(files)
        if use_pct_of_files != 100:
            use_files = int((use_pct_of_files/100.0) * len(self.filenames))
            self.filenames = self.filenames[:use_files]
            get_logger().info('Will use %s%% of batch files, that is %d.', use_pct_of_files, len(self.filenames))

        self.n_files = len(self.filenames)

        self.i = 0
        self.initial_cursor = 0

        self.passes = passes
        self.minibatch_size = minibatch_size

        self.X = None
        self.positions = None
        self.y = None
        self.current_file_size = 0
        self.cursor = 0
        self.scaler = scaler
        self.shuffle_examples = shuffle_examples
        self.nmz = nmz
        self.reg_bias = reg_bias
        self.use_positions = use_positions
        self.label_correction = label_correction
        self.use_levels = use_levels

        get_logger().debug('self.shuffle_examples = %s', self.shuffle_examples)

    def init_scaler(self, filename=None):
        if filename is None:
            mat = np.load(self.filenames[0])
        else:
            mat = np.load(filename)
        X = mat['patches'].item().todense().astype(np.float32)
        y = mat['labels']
        positions = mat['positions']
        if self.use_positions:
            positions = mat['positions']
            X = np.hstack([X, positions])

        X, y, positions, _ = self.apply_levels(X, y, positions, np.zeros((len(y),)))

        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(X)

    def get_file_length(self, ix):
        return np.load(self.filenames[ix])['labels'].shape[0]

    def rewind_to_minibatch(self, minibatch_i):
        size_index = open(join(os.path.dirname(self.filenames[0]), 'sizes.txt'), 'rt').read()
        size_index = eval(size_index)
        examples_consumed = minibatch_i * self.minibatch_size

        at_file = 0
        cursor = 0
        for file_i in range(self.passes * self.n_files):
            filename = basename(self.filenames[file_i % self.n_files])
            if filename not in size_index:
                continue
            cursor_plus_file = cursor + size_index[filename]
            if cursor <= examples_consumed <= cursor_plus_file:
                at_file = file_i
                break
            cursor = cursor_plus_file

        self.i = at_file
        self.initial_cursor = cursor_plus_file-examples_consumed

    def apply_levels(self, X, y, positions, tags):
        if len(self.use_levels) > 0:
            selected_ix = []
            pos = positions
            sub_starts=[i for i in range(pos.shape[0]-1) if (pos[i,0]==0 and pos[i,1]==0) and not (pos[i+1,0]==0 and pos[i+1,1]==0)]
            img_starts = [i for i in range(pos.shape[0]-1) if (pos[i,0]==0 and pos[i,1]==0) and (pos[i+1,0]==0 and pos[i+1,1]==0)]
            if 'L0' in self.use_levels:
                ix = img_starts
                selected_ix.extend(ix)
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L0.')

            if 'L1' in self.use_levels:
                ix = sub_starts[::3]
                selected_ix.extend(ix)
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L1.')
            if 'L2' in self.use_levels:
                ix = sub_starts[1::3]
                selected_ix.extend(ix)
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L2.')
            if 'L3' in self.use_levels:
                ix = sub_starts[2::3]
                selected_ix.extend(ix)
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L3.')

            if len(self.use_levels) == 1:
                tags = np.ones((len(tags),))

            # if self.shuffle_examples:
            #     random.shuffle(selected_ix)
            # else:
            selected_ix = sorted(selected_ix)

            return X[selected_ix, :], y[selected_ix], positions[selected_ix, :], tags[selected_ix]
        else:
            return X, y, positions, tags

    def loader(self):
        if self.cursor >= self.current_file_size: # Depleted current file, will read a new one.
            if self.i >= self.passes*self.n_files: # Or, if no files left to feed just pass None, None, None.
                return None, None, None

            if 0 == (self.i % self.n_files):
                get_logger().info('Pass: %s', (self.i / self.n_files)+1)

            mat = None
            while mat is None:
                try:
                    mat = np.load(self.filenames[self.i % self.n_files])
                except:
                    mat = None
                    self.i += 1
                    get_logger().warn('Could not process file: %s', self.filenames[self.i % self.n_files])

            self.X = mat['patches'].item()
            self.positions = mat['positions']
            self.y = mat['labels'].ravel().astype(dtype=np.uint32)+self.label_correction
            pos = self.positions
            self.tags = np.array([ int((pos[j, 0] == 0 and pos[j, 1] == 0) and (pos[j+1, 0] == 0 and pos[j+1, 1] == 0)) for j in range(pos.shape[0]-1)] + [0])

            (self.X, self.y, self.positions, self.tags) = self.apply_levels(self.X, self.y, self.positions, self.tags)

            self.randix = range(self.X.shape[0])

            if self.shuffle_examples:
                random.shuffle(self.randix)

            self.current_file_size = self.X.shape[0]

            if self.initial_cursor > 0:
                self.cursor = self.initial_cursor
                self.initial_cursor = 0
            else:
                self.cursor = 0
            self.i += 1

        # Getting next minibatch from current file
        start = self.cursor
        end = min(self.cursor+self.minibatch_size, self.current_file_size)
        inc = end-start

        # Rounding up till the end of current NBNL example
        if (not self.shuffle_examples) and (end < len(self.tags)):
            try:
                ix = np.where(self.tags[end:])[0]
                if len(ix) > 0:
                    roundup = ix[0]
                    inc = (end-start)+roundup
                    end += roundup
                else:
                    end = len(self.tags)
            except:
                import pdb; pdb.set_trace()

        X = self.X[ self.randix[start:end], :].todense()
        positions = self.positions[self.randix[start:end], :]
        if self.use_positions:
            X = np.hstack([X, positions])
        y = self.y[self.randix[start:end]]
        tags = self.tags[self.randix[start:end]]
        self.cursor += inc

        # Normalizing the data
        X = X.astype(np.float32)

        if self.nmz != 'no':
            self.scaler.transform(X)

        if self.reg_bias:
            X = add_reg_bias(X) # Regularized bias term

        #print 'Loader:: Minibatch contains %s NBNL examples.' % tags.sum()

        return X.T, y, tags


class ILSVRCPatchMinibatchLoader_Levels(object):

    def __init__(self, files, passes, use_levels, minibatch_size=1024, scaler=None, use_positions=True, nmz='std',  label_correction=-1, shuffle_examples=True):
        self.filenames = sorted(files)
        self.n_files = len(self.filenames)
        self.i = 0
        self.initial_cursor = 0
        self.passes = passes
        self.use_levels = use_levels
        self.minibatch_size = minibatch_size
        self.scaler = scaler
        self.use_positions = use_positions
        self.nmz = nmz
        self.label_correction = label_correction
        self.shuffle_examples = shuffle_examples
        self.filenames = files

        self.X = None
        self.positions = None
        self.y = None
        self.current_file_size = 0
        self.cursor = 0
        self.active_level_i = 0
        self.pass_i = 0


    def init_scaler(self, filename=None):
        if filename is None:
            mat = np.load(self.filenames[0])
        else:
            mat = np.load(filename)
        X = mat['patches'].item().todense().astype(np.float32)
        y = mat['labels']
        positions = mat['positions']
        if self.use_positions:
            positions = mat['positions']
            X = np.hstack([X, positions])

        #X, y, positions = self.apply_levels(X, y, positions)

        self.scaler = StandardScaler(copy=False)
        self.scaler.fit_transform(X)

    def loader(self):
        X, y, tags = self.subloader()
        if len(self.use_levels) == 0:
            return X, y, tags

        if X is None:
            get_logger().debug('Pass finished. Advancing active level.')
            self.active_level_i += 1
            if self.active_level_i >= len(self.use_levels):
                get_logger().debug('No levels left. Finishing pass.')
                self.pass_i += 1
                self.active_level_i = 0
                if self.pass_i >= self.passes:
                    get_logger().debug('No passes remain. Finishing.')
                    return None, None, None

            self.X = None
            self.positions = None
            self.y = None
            self.current_file_size = 0
            self.cursor = 0
            self.i = 0

            X, y, tags = self.subloader()
            assert X is not None

        return X, y, tags

    def apply_levels(self, X, y, positions):
        if len(self.use_levels) > 0:
            selected_ix = []
            pos = positions
            sub_starts=[i for i in range(pos.shape[0]-1) if (pos[i,0]==0 and pos[i,1]==0) and not (pos[i+1,0]==0 and pos[i+1,1]==0)]
            img_starts = [i for i in range(pos.shape[0]-1) if (pos[i,0]==0 and pos[i,1]==0) and (pos[i+1,0]==0 and pos[i+1,1]==0)]
            #import pdb; pdb.set_trace()
            active_levels = [self.use_levels[self.active_level_i]]
            if 'L0' in active_levels:
                selected_ix.extend(img_starts)
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L0.')
            if 'L1' in active_levels:
                selected_ix.extend(sub_starts[::3])
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L1.')
            if 'L2' in active_levels:
                selected_ix.extend(sub_starts[1::3])
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L2.')
            if 'L3' in active_levels:
                selected_ix.extend(sub_starts[2::3])
                get_logger().debug('ILSVRCPatchMinibatchLoader::loader:: Using descriptors from level L3.')

            return X[selected_ix, :], y[selected_ix], positions[selected_ix, :]
        else:
            return X, y, positions

    def subloader(self):
        if self.cursor >= self.current_file_size: # Depleted current file, will read a new one.
            if self.i >= self.n_files: # Or, if no files left to feed just pass None, None, None.
                return None, None, None

            if 0 == (self.i % self.n_files):
                get_logger().info('Pass: %s', (self.i / self.n_files)+1)

            mat = None
            while mat is None:
                try:
                    mat = np.load(self.filenames[self.i % self.n_files])
                except:
                    mat = None
                    self.i += 1
                    get_logger().warn('Could not process file: %s', self.filenames[self.i % self.n_files])

            self.X = mat['patches'].item()
            self.positions = mat['positions']
            self.y = mat['labels'].ravel().astype(dtype=np.uint32)+self.label_correction

            (self.X, self.y, self.positions) = self.apply_levels(self.X, self.y, self.positions)

            pos = self.positions
            self.tags = np.array([ int((pos[j, 0] == 0 and pos[j, 1] == 0) and (pos[j+1, 0] == 0 and pos[j+1, 1] == 0)) for j in range(pos.shape[0]-1)] + [0])
            self.randix = range(self.X.shape[0])

            if self.shuffle_examples:
                random.shuffle(self.randix)

            self.current_file_size = self.X.shape[0]

            if self.initial_cursor > 0:
                self.cursor = self.initial_cursor
                self.initial_cursor = 0
            else:
                self.cursor = 0
            self.i += 1

        # Getting next minibatch from current file
        start = self.cursor
        end = min(self.cursor+self.minibatch_size, self.current_file_size)
        inc = end-start

        # Rounding up till the end of current NBNL example
        if (not self.shuffle_examples) and (end < len(self.tags)):
            try:
                ix = np.where(self.tags[end:])[0]
                if len(ix) > 0:
                    roundup = ix[0]
                    inc = (end-start)+roundup
                    end += roundup
                else:
                    end = len(self.tags)
            except:
                import pdb; pdb.set_trace()

        X = self.X[ self.randix[start:end], :].todense()
        positions = self.positions[self.randix[start:end], :]
        if self.use_positions:
            X = np.hstack([X, positions])
        y = self.y[self.randix[start:end]]
        tags = self.tags[self.randix[start:end]]
        self.cursor += inc

        # Normalizing the data
        X = X.astype(np.float32)

        if self.nmz != 'no':
            self.scaler.transform(X)

        #print 'Loader:: Minibatch contains %s NBNL examples.' % tags.sum()

        return X.T, y, tags

