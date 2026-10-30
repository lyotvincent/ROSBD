import numpy as np
import tifffile
import zarr
import openslide
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def norm_tiles(tiles):
    tiles = (tiles-tiles.min())/(tiles.max()-tiles.min())*255
    tiles = tiles.astype(np.uint8)
    return tiles


# def norm_tiles(tiles, value_range=None):
#     """Map the dtype range to [0, 255] as float32, without integer quantization.

#     Args:
#         tiles: Non-empty real array [N_c, H, W] or AF reference [H, W].
#         value_range: Optional explicit (low, high) override. By default use
#             np.iinfo(dtype).min/max for integers and (0, 1) for bool.
#             Floating-point inputs require an explicit intensity range because
#             their dtype does not identify the original image encoding.

#     Returns:
#         float32 array with the input shape. Input values are not modified.
#         Arrays with the same integer dtype use the same mapping, independent
#         of observed intensities. Values outside an explicit range are clipped.
#     """
#     tiles = np.asarray(tiles)
#     if tiles.ndim not in (2, 3) or any(size == 0 for size in tiles.shape):
#         raise ValueError("tiles must be a non-empty [H, W] or [N_c, H, W] array")
#     if tiles.dtype.kind not in "biuf":
#         raise ValueError("tiles must contain real numeric intensities")
#     values = tiles.astype(np.float64)
#     if not np.isfinite(values).all():
#         raise ValueError("tiles must contain only finite intensities")

#     if value_range is None:
#         if np.issubdtype(tiles.dtype, np.integer):
#             limits = np.iinfo(tiles.dtype)
#             low, high = float(limits.min), float(limits.max)
#         elif tiles.dtype.kind == "b":
#             low, high = 0.0, 1.0
#         else:
#             raise ValueError("Floating-point tiles require an explicit value_range")
#     else:
#         bounds = np.asarray(value_range, dtype=np.float64)
#         if bounds.shape != (2,) or not np.isfinite(bounds).all():
#             raise ValueError("value_range must contain two finite numbers (low, high)")
#         low, high = map(float, bounds)
#         if high <= low:
#             raise ValueError("value_range high must be greater than low")

#     normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
#     return (normalized * 255.0).astype(np.uint8)


def read_tile(slide_reader: openslide.OpenSlide, coord, tile_size):
    """Read a grayscale tile at level-0 (x, y), with tile_size (H, W)."""
    height, width = tile_size
    tile = slide_reader.read_region(tuple(coord), 0, (width, height)).convert("L")
    return tile


def read_region(slide_reader: openslide.OpenSlide, coords, tile_size, fill=False):
    """
    Read a region consisting of multiple tiles at level 0.

    Args:
        slide_reader: OpenSlide reader.
        coords: Integer array [N, 2] of tile top-left pixel coordinates (x, y).
        tile_size: (H, W), in pixels.
        fill: If True, read the entire bounding rectangle, including gaps.
            Otherwise, read only the specified tiles and fill gaps with zeros.

    Returns:
        region: float32 grayscale array [H, W] in [0, 255], starting at
            the minimum (x, y), normalized using each source tile's dtype.

    Raises:
        ValueError: If coords is empty or not an integer array [N, 2], or
            tile_size does not contain two positive integers.
    """
    coords = np.asarray(coords)
    tile_size = np.asarray(tile_size)

    tile_height, tile_width = map(int, tile_size)
    min_x, min_y = map(int, coords.min(axis=0))
    max_x, max_y = map(int, coords.max(axis=0))
    height = max_y - min_y + tile_height
    width = max_x - min_x + tile_width

    if fill:
        return norm_tiles(np.asarray(read_tile(slide_reader, (min_x, min_y), (height, width))))

    region = np.zeros((height, width), dtype=np.float32)
    for x, y in coords:
        x, y = int(x), int(y)
        # tile = norm_tiles(np.asarray(read_tile(slide_reader, (x, y), (tile_height, tile_width))))
        tile = np.asarray(read_tile(slide_reader, (x, y), (tile_height, tile_width)))
        row, col = y - min_y, x - min_x
        region[row:row + tile_height, col:col + tile_width] = tile
    return region


def read_multichannel_tiles(mutialchannel_dict: dict, coord, tile_size):
    """Read level-0 TIFF channel tiles in their original intensity units.

    Args:
        mutialchannel_dict: Non-empty mapping of channel names to TIFF paths or (path, page_index).
            Channels are returned in mapping insertion order.
        coord: Integer top-left pixel coordinate (x, y).
        tile_size: Positive integer (H, W), in pixels.

    Returns:
        tiles: array [N_c, H, W], retaining native intensities and source dtype.
            Out-of-image pixels are zero. No intensity normalization is applied.

    Reads the first full-resolution, two-dimensional grayscale TIFF page via
    a Zarr view, decoding only the strips/tiles needed for the requested area.
    """

    coord = np.asarray(coord)
    tile_size = np.asarray(tile_size)

    x, y = map(int, coord)
    height, width = map(int, tile_size)
    tiles = []
    for channel_path in mutialchannel_dict.values():
        # COMET channels reference separate pages of a shared multichannel TIFF.
        page_index = 0
        if isinstance(channel_path, tuple):
            channel_path, page_index = channel_path
        with tifffile.TiffFile(channel_path) as slide:
            page = slide.series[0].pages[page_index]
            if page.ndim != 2:
                raise ValueError(f"Expected a 2-D grayscale TIFF page: {channel_path}")
            image_height, image_width = page.shape
            tile = np.zeros((height, width), dtype=page.dtype)
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + width, image_width), min(y + height, image_height)
            if x0 < x1 and y0 < y1:
                with page.aszarr() as store:
                    pixels = zarr.open(store, mode="r")
                    tile[y0 - y:y1 - y, x0 - x:x1 - x] = pixels[y0:y1, x0:x1]
            # tiles.append(norm_tiles(tile))
            tiles.append(tile)
    return np.stack(tiles, axis=0)


def show_multichannel_tiles(channels_names, tiles, ncol=3):
    """Show grayscale channel tiles with an automatically calculated row count.

    Args:
        channels_names: Channel names in the same order as tiles.
        tiles: Non-empty array [N_c, H, W].
        ncol: Positive number of columns.

    Returns:
        figures: A one-element list containing the matplotlib Figure.
            Unused axes in the last row are hidden.
    """
    # tiles = np.asarray(tiles, dtype=np.uint8)
    tiles = norm_tiles(tiles)
    channels_names = list(channels_names)
    if tiles.ndim != 3 or any(size == 0 for size in tiles.shape):
        raise ValueError("tiles must be a non-empty array with shape (N_c, H, W)")
    if len(channels_names) != tiles.shape[0]:
        raise ValueError("channels_names must contain one name per channel")
    if (
        isinstance(ncol, (bool, np.bool_))
        or not isinstance(ncol, (int, np.integer))
        or ncol <= 0
    ):
        raise ValueError("ncol must be a positive integer")

    nrow = (len(tiles) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False
    )
    for index, ax in enumerate(axes.flat):
        if index < len(tiles):
            ax.imshow(tiles[index], cmap="gray")
            ax.set_title(str(channels_names[index]))
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return [fig]


def show_multichannel_tiles_compare(channels_names, tiles_left, tiles_right, ncol=3):
    """Show matching channels from two tile groups side by side.

    Args:
        channels_names: Channel names in the same order as both tile groups.
        tiles_left: Non-empty array [N_c, H, W], displayed on the left.
        tiles_right: Array with the same shape, displayed on the right.
        ncol: Positive number of channel comparisons per row. The row count
            is calculated automatically.

    Returns:
        figures: A one-element list containing the matplotlib Figure.
            Each subplot has one channel title and a concatenated [H, 2W]
            grayscale image, with a shared intensity scale for both halves.
    """
    tiles_left = norm_tiles(tiles_left)
    tiles_right = norm_tiles(tiles_right)
    # tiles_left = np.asarray(tiles_left)
    # tiles_right = np.asarray(tiles_right)
    if tiles_left.ndim != 3 or any(size == 0 for size in tiles_left.shape):
        raise ValueError("tiles_left must be a non-empty array with shape (N_c, H, W)")
    if tiles_right.shape != tiles_left.shape:
        raise ValueError("tiles_left and tiles_right must have the same shape")

    tiles = np.concatenate((tiles_left, tiles_right), axis=2)
    return show_multichannel_tiles(channels_names, tiles, ncol=ncol)


def save_tile_outputs(save_dir, coord, channel_names, result, bg, raw, brightness=2.0):
    """Save individual channel tiles under save_dir/channel_name/type/x_y.

    result, bg and raw each have a lossless original-intensity .tif and a
    preview .png. result_raw_combinded contains a .png with result on the
    left and raw on the right. PNG sizes are (H,W), or (H,2W) for comparisons,
    without axes, titles or borders. Preview normalization remains shared
    across the channel stack for each output type, as before.

    brightness: Non-negative finite PNG intensity multiplier. 1 keeps the
        existing preview, >1 brightens it, <1 darkens it. Values are clipped
        to [0,255]. The same multiplier applies to all previews and both
        comparison halves. TIFF intensities and input arrays are unchanged.
    """

    try:
        brightness = float(brightness)
    except (TypeError, ValueError):
        raise ValueError("brightness must be a finite non-negative number") from None
    if not np.isfinite(brightness) or brightness < 0:
        raise ValueError("brightness must be a finite non-negative number")

    channel_names = [str(name) for name in channel_names]
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique")
    if any(not name or name in (".", "..") or "/" in name or "\\" in name
           for name in channel_names):
        raise ValueError("Each channel name must be a single directory name")
    original_images = {
        "result": np.asarray(result), "bg": np.asarray(bg), "raw": np.asarray(raw)
    }
    if any(image.ndim != 3 or image.shape[0] != len(channel_names)
           or any(size == 0 for size in image.shape) for image in original_images.values()):
        raise ValueError("result, bg and raw must be non-empty [N_c,H,W] arrays matching channel_names")
    if len({image.shape for image in original_images.values()}) != 1:
        raise ValueError("result, bg and raw must have the same shape")

    previews = {
        name: np.clip(norm_tiles(image), 0, 255).astype(np.uint8) if image.max() != image.min()
        else np.zeros(image.shape, dtype=np.uint8)
        for name, image in original_images.items()
    }
    previews = {
        name: np.clip(image.astype(np.float64) * brightness, 0, 255).astype(np.uint8)
        for name, image in previews.items()
    }
    x, y = map(int, coord)
    for index, channel_name in enumerate(channel_names):
        channel_dir = Path(save_dir) / channel_name
        for name, image in original_images.items():
            directory = channel_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                directory / f"{x}_{y}.tif", image[index],
                photometric="minisblack", compression="deflate",
                metadata={"axes": "YX", "channel_name": channel_name},
            )
            Image.fromarray(previews[name][index]).save(directory / f"{x}_{y}.png")
        paired_dir = channel_dir / "result_raw_combinded"
        paired_dir.mkdir(parents=True, exist_ok=True)
        paired = np.concatenate((previews["result"][index], previews["raw"][index]), axis=1)
        Image.fromarray(paired).save(paired_dir / f"{x}_{y}.png")



def select_samples_to_process(sample_ids, save_dir, overwrite=False):
    """Skip samples with an existing output path unless overwrite is enabled."""
    pending = []
    for sample_id in sample_ids:
        sample_dir = Path(save_dir) / sample_id
        if not overwrite and sample_dir.exists():
            print(f"Skipping existing sample: {sample_dir}", flush=True)
            continue
        pending.append(sample_id)
    return pending
