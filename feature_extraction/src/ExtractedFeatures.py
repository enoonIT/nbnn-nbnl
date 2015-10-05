import numpy as np

class ExtractedFeatures:
    def __init__(self, num_items, dim):
        self.patches6 = np.zeros( (num_items, dim), dtype='float32' )
        self.patches7 = np.zeros( (num_items, dim), dtype='float32' )
        self.pos = np.zeros( (num_items, 2), dtype='uint16' )
        self.cursor = 0

    def append(self, features6, features7, pos):
        self.patches6[self.cursor:self.cursor+features6.shape[0], :] = features6
        self.patches7[self.cursor:self.cursor+features7.shape[0], :] = features7
        self.pos[self.cursor:self.cursor+pos.shape[0], :] = pos
        self.cursor += features6.shape[0]
        assert features6.shape==features7.shape, "Size mismatch between features"
        assert features6.shape[0]==pos.shape[0], "Size mismatch between features and positions"

    def get(self):
        print "Was " + str(self.patches6.shape)
        self.patches6.resize( (self.cursor, self.patches6.shape[1]) ) #Throws away unused patches
        self.patches7.resize( (self.cursor, self.patches7.shape[1]) )
        self.pos.resize( (self.cursor, self.pos.shape[1]) )
        print "Is " + str(self.patches6.shape)
        return (self.patches6, self.patches7, self.pos)