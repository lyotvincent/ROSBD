"""Multichannel array wrapper for the official CLEAR-AF file API.
Source: https://github.com/antoranzlab/CLEAR-AF
Upstream commit: a681dc670ed4600286e87b09600c110459e86a31
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import warnings


def clear_af_2d(tiles, af_ref):
    """Correct each channel using the same aligned AF reference.

    Args:
        tiles: Non-empty mixed-signal array [N_c, H, W].
        af_ref: Autofluorescence reference array [H, W], aligned to tiles.

    Returns:
        result: Corrected uint16 array [N_c, H, W], in input channel order.
        bg: Removed signal, tiles - result, as float64 [N_c, H, W].
            Includes the effects of upstream suppression and rounding; this
            is the effective removed background, not the internal AF estimate.

    Uses all upstream defaults, including the bundled medoids. Input
    intensities are passed unchanged through temporary TIFFs to the official
    single-channel API as float64, avoiding NumPy's uint8-to-float16
    arctan2 promotion, which SciPy's uniform_filter1d does not support.
    Intensity values are preserved without normalization. Channels without
    any jointly positive MS/AF pixels return zero corrected signal, matching
    upstream's final foreground-mask rule without fitting an empty histogram.
    A RuntimeWarning identifies each such channel index.
    """
    tiles = np.asarray(tiles)
    af_ref = np.asarray(af_ref)
    if tiles.ndim != 3 or any(size == 0 for size in tiles.shape):
        raise ValueError("tiles must be a non-empty array with shape (N_c, H, W)")
    if af_ref.ndim != 2 or af_ref.shape != tiles.shape[1:]:
        raise ValueError("af_ref must have shape (H, W) matching tiles")

    if not np.isfinite(tiles).all() or not np.isfinite(af_ref).all():
        raise ValueError("CLEAR-AF inputs must contain finite intensities")

    import tifffile
    from clearaf import autofluorescence_subtraction

    with TemporaryDirectory(prefix="clearaf_") as directory:
        root = Path(directory)
        ms, af, ts, qc = [root / name for name in ("ms.tif", "af.tif", "ts.tif", "qc.tif")]
        tifffile.imwrite(af, af_ref.astype(np.float64), photometric="minisblack")
        corrected = []
        for index, tile in enumerate(tiles):
            if not np.any((tile > 0) & (af_ref > 0)):
                warnings.warn(
                    f"CLEAR-AF channel index {index}: no jointly positive MS/AF "
                    "pixels; returning zero corrected signal per foreground mask",
                    RuntimeWarning, stacklevel=2,
                )
                corrected.append(np.zeros(tile.shape, dtype=np.uint16))
                continue
            tifffile.imwrite(ms, tile.astype(np.float64), photometric="minisblack")
            autofluorescence_subtraction(str(ms), str(af), str(ts), str(qc))
            corrected.append(tifffile.imread(ts))
    result = np.stack(corrected, axis=0)
    bg = tiles.astype(np.float64) - result.astype(np.float64)
    return result, bg
