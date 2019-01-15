class BreakoutGoalFeatureExtractor(FeatureExtractor):
    def __init__(self, obs_space, bricks_rows=6, bricks_cols=18):
        self.bricks_rows = bricks_rows
        self.bricks_cols = bricks_cols
        output_space = Box(low=0, high=1, shape=(bricks_cols, bricks_rows), dtype=np.uint8)
        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        bricks_features = np.ones((self.bricks_cols, self.bricks_rows))
        for row, col in itertools.product(range(self.bricks_rows), range(self.bricks_cols)):
            # Pixel of the observation to check:
            px_upper_left  = int( 8 + 8 * col)
            py_upper_left  = int(57 + 6 * row)
            px_upper_right = int(15 + 8 * col)
            py_upper_right = int(57 + 6 * row)

            # Checking max because the input has 3 channels:
            if max(input[py_upper_left][px_upper_left]) == 0 or \
                max(input[py_upper_right][px_upper_right]) == 0:
                bricks_features[col][row] = 0

        return bricks_features
