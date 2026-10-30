import numpy as np
from skimage.restoration import denoise_tv_chambolle


def shared_background_filter(
    img,
    q_b=0.40,
    q_a=0.40,
    scale_percentile=99.5,
    clip_max=2.0,
    tv_weight=0.05,
    max_iter=50,
    tol=1e-4,
    min_b=1e-4,
    eps=1e-6,
):
    """
    Rank-1 multi-channel shared background decomposition.

    Model
    -----
    For each pixel p:

        y_p = b_p * a + s_p

    where:
        y_p : observed C-channel intensity vector
        b_p : shared background amplitude at pixel p
        a   : learned C-channel background coefficient vector
        s_p : marker-specific residual signal

    Parameters
    ----------
    img : np.ndarray
        Shape [C, H, W], e.g. [7, 256, 256].
        Can be uint8 / uint16 / float.

    q_b : float
        Quantile used to estimate pixel-wise background b.
        Default 0.20.

    q_a : float
        Quantile used to update channel background coefficients a.
        Default 0.40.

    scale_percentile : float
        Per-channel robust normalization percentile.
        Default 99.5.

    clip_max : float or None
        Clip normalized intensities only during parameter fitting.
        Strong marker signals above this value will not dominate fitting.
        Set None to disable.

    tv_weight : float
        TV smoothing strength for spatial background map.
        Set 0 to disable spatial smoothing.

    max_iter : int
        Maximum number of alternating iterations.

    tol : float
        Convergence tolerance for a.

    min_b : float
        Ignore pixels with extremely low estimated background
        when updating a.

    eps : float
        Numerical stability constant.

    Returns
    -------
    img_filtered : np.ndarray
        Shape [N_c, H, W], float32.
        Filtered image in the ORIGINAL intensity scale.

    params : dict
        Learned / estimated parameters and intermediate results.

    pollution_predicted : np.ndarray
        Shape [C, H, W], float32.
        Predicted pollution for each channel in the ORIGINAL intensity scale.
    """
    img = np.asarray(img)

    if img.ndim != 3:
        raise ValueError(f"Expected img shape [N_c, H, W], got {img.shape}")

    img_raw = img.astype(np.float32, copy=False)
    C, H, W = img_raw.shape

    # ------------------------------------------------------------
    # 1. Robust channel-wise normalization
    # ------------------------------------------------------------
    channel_scales = np.ones(C, dtype=np.float32)

    for c in range(C):
        values = img_raw[c].reshape(-1)
        positive = values[values > 0]

        if len(positive) > 0:
            scale = np.percentile(positive, scale_percentile)
        else:
            scale = 1.0

        if not np.isfinite(scale) or scale < eps:
            scale = 1.0

        channel_scales[c] = scale

    img_norm = img_raw / channel_scales[:, None, None]
    # Keep original normalized image for final subtraction.
    img_fit = img_norm.copy()
    # Clip ONLY during parameter estimation.
    if clip_max is not None:
        img_fit = np.clip(img_fit, 0, clip_max)

    # ------------------------------------------------------------
    # 2. Convert:
    #
    # [N_c, H, W]
    #       ↓
    # [N_pixel, N_c]
    # ------------------------------------------------------------
    Y = np.moveaxis(img_fit, 0, -1).reshape(-1, C)
    # Ignore completely empty pixels during parameter learning
    pixel_valid = np.max(Y, axis=1) > eps
    Y_fit = Y[pixel_valid]

    if len(Y_fit) == 0:
        raise ValueError("All pixels are zero. Cannot estimate background.")

    # ------------------------------------------------------------
    # 3. Initialize channel background vector a
    #
    # a = [a1, a2, ..., ac]
    # ------------------------------------------------------------
    a = np.ones(C, dtype=np.float32)
    history = []
    converged = False

    # ------------------------------------------------------------
    # 4. Alternating estimation
    # ------------------------------------------------------------
    for iteration in range(max_iter):

        # ========================================================
        # Step A:
        #
        # y_pc = b_p * a_c + s_pc
        #
        # therefore
        #
        # y_pc / a_c >= b_p
        #
        # Estimate b_p using low cross-channel quantile.
        # ========================================================

        ratio = Y_fit / (a[None, :] + eps)
        b = np.quantile(ratio, q_b, axis=1).astype(np.float32)
        # Pixels with almost zero b are not useful
        # for estimating a.
        valid_b = np.isfinite(b) & (b > min_b)
        if valid_b.sum() < 10:
            break

        # ========================================================
        # Step B:
        #
        # y_pc / b_p
        #     =
        # a_c + s_pc / b_p
        #
        # Signal pushes this value upward,
        # so use a relatively low quantile.
        # ========================================================
        a_new = np.empty(C, dtype=np.float32)
        for c in range(C):
            values = Y_fit[valid_b, c] / (b[valid_b] + eps)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                a_new[c] = a[c]
            else:
                a_new[c] = np.quantile(values, q_a)

        # Prevent numerical collapse
        a_new = np.maximum(a_new, eps)

        # --------------------------------------------------------
        # Resolve scale ambiguity:
        #
        # b * a == (k*b) * (a/k)
        #
        # Fix median(a) = 1.
        # --------------------------------------------------------
        a_median = np.median(a_new)
        if not np.isfinite(a_median) or a_median < eps:
            a_median = 1.0

        a_new = a_new / a_median
        # convergence
        diff = np.mean(np.abs(a_new - a))
        history.append(
            {
                "iteration": iteration + 1,
                "diff": float(diff),
                "a": a_new.copy(),
            }
        )

        a = a_new
        if diff < tol:
            converged = True
            break

    # ------------------------------------------------------------
    # 5. Estimate full-resolution background map b(x, y)
    #
    # Important:
    # use UNCLIPPED normalized image here.
    # ------------------------------------------------------------
    ratio_full = img_norm / (a[:, None, None] + eps)
    b_map = np.quantile(ratio_full, q_b, axis=0).astype(np.float32)
    # Empty spatial pixels
    spatial_valid = np.max(img_norm, axis=0) > eps
    b_map[~spatial_valid] = 0

    # ------------------------------------------------------------
    # 6. Spatial regularization
    # ------------------------------------------------------------
    if tv_weight is not None and tv_weight > 0:
        b_smooth = denoise_tv_chambolle(
            b_map, weight=tv_weight, channel_axis=None
        ).astype(np.float32)
        # Avoid spreading background into completely empty region
        b_smooth[~spatial_valid] = 0

    else:
        b_smooth = b_map.copy()

    # Background should not become negative
    b_smooth = np.maximum(b_smooth, 0)

    # ------------------------------------------------------------
    # 7. Reconstruct background for each marker
    #
    # B_c(x,y) = a_c * b(x,y)
    # ------------------------------------------------------------
    background_norm = a[:, None, None] * b_smooth[None, :, :]

    # Because signal S >= 0:
    #
    # background <= observed intensity
    #
    # TV smoothing can slightly violate this constraint,
    # therefore enforce it explicitly.
    background_norm = np.minimum(background_norm, img_norm)
    background_norm = np.maximum(background_norm, 0)

    # ------------------------------------------------------------
    # 8. Remove shared background
    # ------------------------------------------------------------
    filtered_norm = img_norm - background_norm
    filtered_norm = np.maximum(filtered_norm, 0)

    # ------------------------------------------------------------
    # 9. Convert back to ORIGINAL intensity scale
    # ------------------------------------------------------------
    img_filtered = (filtered_norm * channel_scales[:, None, None]).astype(np.float32)
    pollution_predicted = (background_norm * channel_scales[:, None, None]).astype(
        np.float32
    )

    # Relative background coefficients in original raw scale.
    a_raw = a * channel_scales
    a_raw = (a_raw / (np.median(a_raw) + eps)).astype(np.float32)

    # ------------------------------------------------------------
    # 10. Return learned parameters / diagnostic information
    # ------------------------------------------------------------
    params = {
        # Most important learned parameter
        "a_normalized": a.copy(),
        # Relative coefficients converted back toward raw-space
        "a_raw_relative": a_raw,
        # Channel normalization factors
        "channel_scales": channel_scales,
        # Pixel-level latent shared background
        "b_map": b_map,
        # Spatially regularized background field
        "b_smooth": b_smooth,
        # Predicted pollution for each channel
        "background_estimate": pollution_predicted,
        # Optimization information
        "n_iter": len(history),
        "converged": converged,
        "final_diff": (history[-1]["diff"] if len(history) > 0 else None),
        "history": history,
        # Hyperparameters
        "q_b": q_b,
        "q_a": q_a,
        "scale_percentile": scale_percentile,
        "clip_max": clip_max,
        "tv_weight": tv_weight,
    }

    return img_filtered, params, pollution_predicted
