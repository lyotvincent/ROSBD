"""Sauvola foreground masking via scikit-image.
Source: https://github.com/scikit-image/scikit-image
"""

import numpy as np
from skimage.filters import threshold_sauvola


def sauvola_2d(img, window_size=15, k=0.2, r=None, return_mask=False):
    """Keep original intensities above a local Sauvola threshold in [H, W].

    window_size is odd, in pixels. r=None uses half the input dtype range,
    following scikit-image. For floats in raw intensity units, supply r explicitly
    (e.g. 32767.5 for uint16-scale data). Shape and dtype are preserved.
    Returns the image, or (image, boolean mask) with return_mask=True.
    """
    img = np.asarray(img)
    threshold = threshold_sauvola(img, window_size=window_size, k=k, r=r)
    mask = img > threshold
    filtered = np.where(mask, img, 0)
    return (filtered, mask) if return_mask else filtered
