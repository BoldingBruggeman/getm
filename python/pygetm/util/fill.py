import logging
from typing import Optional

import numpy as np
import scipy.spatial


class Filler:
    def __init__(
        self,
        mask: np.ndarray,
        dim_weights=None,
        logger: Optional[logging.Logger] = None,
    ):
        logging.basicConfig(level=logging.INFO)
        logger = logger or logging.getLogger()

        self.masked = mask
        self.unmasked = ~mask
        coords = np.moveaxis(np.indices(mask.shape, dtype=float), 0, -1)
        if dim_weights is not None:
            assert len(dim_weights) == mask.ndim
            coords *= dim_weights
        logger.debug("  - building KDTree")
        tree = scipy.spatial.cKDTree(coords[self.unmasked, :])
        logger.debug("  - finding nearest neighbors")
        _, self.inearest = tree.query(coords[self.masked, :], workers=-1)

    def __call__(self, values):
        values[self.masked] = values[self.unmasked][self.inearest]
