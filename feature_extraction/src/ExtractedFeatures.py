class ExtractedFeatures:
    def __init__(self, num_items, dim):
        self.patches6 = np.zeros( (num_items, dim), dtype='float' )
        self.patches7 = np.zeros( (num_items, dim), dtype='float' )
        self.pos = np.zeros( (num_items, 2), dtype='float' )
        self.cursor = 0

    def append(self, features6, features7, pos):
        self.patches6[self.cursor:self.cursor+features6.shape[0], :] = features6
        self.patches7[self.cursor:self.cursor+features7.shape[0], :] = features7
        if (pos.shape[0] == 1) and (features6.shape[0]):
            # If there's mismatch in number of features and positions,
            # replicating positions
            pos = np.tile(pos, (features6.shape[0], pos.shape[0]) )

        self.pos[self.cursor:self.cursor+pos.shape[0], :] = pos
        self.cursor += features.shape[0]

    def get(self):
        self.patches6.resize( (self.cursor, self.patches6.shape[1]) )
        self.patches7.resize( (self.cursor, self.patches7.shape[1]) )
        self.pos.resize( (self.cursor, self.pos.shape[1]) )

        return (self.patches6, self.patches7, self.pos)