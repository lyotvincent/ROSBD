from PIL import Image
import cv2
import numpy as np
import os
import h5py
import openslide

def remove_small_components(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    output = np.zeros_like(mask)

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            output[labels == label_id] = 255

    return output


def clean_tissue_mask(tissue_mask, open_kernel_size=3, close_kernel_size=15):
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )

    clean_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    return clean_mask


def otsu_tissue_mask(lowres_img):
    hsv = cv2.cvtColor(lowres_img, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]

    threshold, tissue_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return saturation, tissue_mask, threshold


def get_lowres_wsi(wsi, target_size=3000):
    dimensions = wsi.level_dimensions
    max_sizes = [max(width, height) for width, height in dimensions]
    level = min(range(wsi.level_count), key=lambda i: abs(max_sizes[i] - target_size))
    width, height = dimensions[level]

    img = wsi.read_region((0, 0), level, (width, height)).convert("RGB")

    img = np.array(img)

    width0, height0 = wsi.level_dimensions[0]

    scale_x = width0 / width
    scale_y = height0 / height

    # print(f"Selected level: {level}")
    # print(f"Level 0 size: {width0} × {height0}")
    # print(f"Low-res size: {width} × {height}")
    # print(f"Scale: x={scale_x:.3f}, y={scale_y:.3f}")

    return img, level, scale_x, scale_y


def griding_wsi(
    wsi,
    patch_size,
    overlap,
    tissue_mask,
    scale_x,
    scale_y,
    min_tissue_ratio=0.4,
):
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= patch_size:
        raise ValueError("overlap must be smaller than patch_size")
    stride = patch_size - overlap
    width, height = wsi.dimensions

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # ===============================
            # tissue filtering
            # ===============================
            if tissue_mask is not None:

                tissue_ratio = get_patch_tissue_ratio(
                    x=x,
                    y=y,
                    patch_size=patch_size,
                    tissue_mask=tissue_mask,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    wsi_width=width,
                    wsi_height=height,
                )

                if tissue_ratio < min_tissue_ratio:
                    continue

            read_w = min(patch_size, width - x)
            read_h = min(patch_size, height - y)

            region = wsi.read_region((x, y), 0, (read_w, read_h)).convert("RGB")

            if read_w == patch_size and read_h == patch_size:
                patch = region

            else:
                patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))

                patch.paste(region, (0, 0))

            yield patch, x, y


def get_tissue_mask(
    wsi,
    target_size=5000,
    open_kernel_size=3,
    close_kernel_size=10,
    min_area=50,
):

    # -------------------------
    # 1. low-resolution WSI
    # -------------------------
    lowres_img, level, scale_x, scale_y = get_lowres_wsi(wsi, target_size=target_size)

    # -------------------------
    # 2. HSV Saturation + Otsu
    # -------------------------
    saturation, tissue_mask, threshold = otsu_tissue_mask(lowres_img)

    # -------------------------
    # 3. morphology
    # -------------------------
    tissue_mask = clean_tissue_mask(
        tissue_mask,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
    )

    # -------------------------
    # 4. remove small regions
    # -------------------------

    tissue_mask = remove_small_components(tissue_mask, min_area)

    return {
        "mask": tissue_mask,
        "level": level,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "otsu_threshold": threshold,
    }


def get_patch_tissue_ratio(
    x, y, patch_size, tissue_mask, scale_x, scale_y, wsi_width, wsi_height
):
    x2 = min(x + patch_size, wsi_width)
    y2 = min(y + patch_size, wsi_height)

    mx1 = int(np.floor(x / scale_x))
    my1 = int(np.floor(y / scale_y))

    mx2 = int(np.ceil(x2 / scale_x))
    my2 = int(np.ceil(y2 / scale_y))

    mask_h, mask_w = tissue_mask.shape

    mx1 = np.clip(mx1, 0, mask_w)
    mx2 = np.clip(mx2, 0, mask_w)

    my1 = np.clip(my1, 0, mask_h)
    my2 = np.clip(my2, 0, mask_h)

    region = tissue_mask[my1:my2, mx1:mx2]

    if region.size == 0:
        return 0.0

    tissue_ratio = np.mean(region > 0)

    return float(tissue_ratio)


def stitch_wsi(
    wsi_path,
    h5_path,
    output_path=None,
    patch_size=256,
    target_size=6000,
    coords_key="coords",
    background=(255, 255, 255),
):
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if target_size <= 0:
        raise ValueError("target_size must be > 0")

    with h5py.File(h5_path, "r") as h5_file:
        if coords_key not in h5_file:
            raise KeyError(f"dataset {coords_key!r} was not found in {h5_path}")
        coords = np.asarray(h5_file[coords_key])

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{coords_key!r} must have shape (N, 2), got {coords.shape}")

    slide = openslide.OpenSlide(os.fspath(wsi_path))
    try:
        wsi_width, wsi_height = slide.dimensions
        output_scale = max(wsi_width, wsi_height) / float(target_size)
        output_scale = max(1.0, output_scale)
        output_width = max(1, int(round(wsi_width / output_scale)))
        output_height = max(1, int(round(wsi_height / output_scale)))

        canvas = Image.new("RGB", (output_width, output_height), background)

        read_level = min(
            range(slide.level_count),
            key=lambda level: abs(slide.level_downsamples[level] - output_scale),
        )
        read_downsample = float(slide.level_downsamples[read_level])
        read_size = max(1, int(np.ceil(patch_size / read_downsample)))

        for x, y in coords:
            x, y = int(x), int(y)
            if (
                x >= wsi_width
                or y >= wsi_height
                or x + patch_size <= 0
                or y + patch_size <= 0
            ):
                continue

            left = int(round(x / output_scale))
            top = int(round(y / output_scale))
            right = int(round((x + patch_size) / output_scale))
            bottom = int(round((y + patch_size) / output_scale))
            tile_width = max(1, right - left)
            tile_height = max(1, bottom - top)

            tile = slide.read_region(
                (x, y), read_level, (read_size, read_size)
            ).convert("RGB")
            if tile.size != (tile_width, tile_height):
                tile = tile.resize((tile_width, tile_height), Image.Resampling.BILINEAR)

            # PIL automatically clips patches that touch the output boundary.
            canvas.paste(tile, (left, top))

        if output_path is not None:
            output_path = os.fspath(output_path)
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            canvas.save(output_path)

        return canvas
    finally:
        slide.close()
