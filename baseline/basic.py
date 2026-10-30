"""BaSiC same-channel batch correction via BaSiCPy.
Source: https://github.com/peng-lab/BaSiCPy (tested with 2.0.0).
"""

import numpy as np


def fit_basic(images, get_darkfield=True, working_size=128,
              sort_intensity=True, **model_kwargs):
    """Fit illumination profiles on [N, H, W] images of one channel.

    Use multiple patches from the same source slide and channel. Defaults to
    estimating both flat-field and dark-field, with official approximate fitting.
    Intensity sorting is enabled for independent spatial patches.
    model_kwargs are passed to BaSiC (e.g. device, smoothness_flatfield).
    """
    from basicpy import BaSiC

    model = BaSiC(get_darkfield=get_darkfield, working_size=working_size,
                  sort_intensity=sort_intensity, **model_kwargs)
    model.fit(np.asarray(images))
    return model


def basic_batch(images, model=None, get_darkfield=True, working_size=128,
                sort_intensity=True, return_model=False, **model_kwargs):
    """Correct a [N, H, W] same-channel batch; return nonnegative float32.

    If model is supplied, apply its fitted profiles without refitting.
    Uses (image - darkfield) / flatfield, with is_timelapse=False.
    Returns the batch, or (batch, fitted model) with return_model=True.
    """
    images = np.asarray(images)
    if model is None:
        model = fit_basic(images, get_darkfield=get_darkfield,
                          working_size=working_size, sort_intensity=sort_intensity,
                          **model_kwargs)
    filtered = np.maximum(model.transform(images, use_tqdm=False), 0)
    return (filtered, model) if return_model else filtered
