from numpy_utils import *

class Ray:
    center: np.ndarray
    dir: np.ndarray

    def __init__(self, center: np.ndarray, dir: np.ndarray):
        self.center = center
        self.dir = dir # FIXME normalize
