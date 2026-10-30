# ROSBD image-processing runners

This directory provides image-processing scripts for Rank-One Shared-Background Decomposition (ROSBD) and four baseline methods. ROSBD retains the identifier `sdf` in script names and command-line options.

![overview](overview/overview.png)

## Quick start

Edit the settings in `run_method.sh`, then run:

```sh
sh run_method.sh
```

- `METHOD`: `sdf`, `lmsf`, `sauvola`, `basic`, or `clearaf`.
- `DATASET`: `orion`, `cycif`, `bmcycif`, or `comet`.
- `TASK`: use `single` to process the specified `SAMPLE_ID`, or `cohort` to process all samples in the dataset.
- `WORKER`: number of worker processes. `TILE_SIZE` must match the existing HDF5 coordinate files.
- `SAVE_DIR`: output root directory. With `OVERWRITE=False`, existing sample directories are skipped without checking whether their outputs are complete.
- `FIT_SAMPLES`: maximum number of tiles per channel used to fit BaSiC. For breast TMA, this setting limits the number of complete fields of view instead.

The shell script invokes the multiprocessing runners in this directory, automatically loads the corresponding parameter JSON files, and uses the `wsi` conda environment.

## COMET dataset

COMET is handled by `CometDataset` in `dataset/comet_dataset.py`:

- Source images: `/data/wzx/COMET/COMET_ome_tiff/core*_comet.ome.tif`.
- Sample IDs: `core01`, `core02`, `core03`, and `core04`.
- Coordinates: the `coords` array in `/data/wzx/COMET/patch_info/512_512/<sample_id>.h5`, containing level-0 `(x, y)` pixel coordinates.
- Default channels, as defined in the dataset code: CD20, CD1c, CD206, E-cadherin, FXIIIa, IL-10, Podoplanin, HepPar1, and DAPI. `SCORE_CHANNELS` contains the first eight markers; the runners use `DEFAULT_CHANNELS` for both processing and output. They do not perform cellular classification or scoring.
- Each channel is represented by a `(TIFF path, page index)` pair. Images are read in channel order at their native intensity scale, without per-tile intensity normalization.
- Supported methods: `sdf`, `lmsf`, `sauvola`, and `basic`. Neither COMET nor breast TMA has a configured autofluorescence (AF) reference, so `clearaf` is unavailable for these datasets.
- For COMET, BaSiC is fitted separately for each channel using multiple tiles from the same core. Breast TMA retains its fitting scheme based on complete fields of view.

To process the COMET cohort, set the following values in `run_method.sh`:

```sh
METHOD=sdf
TASK=cohort
DATASET=comet
SAMPLE_ID=core01
TILE_SIZE=512
WORKER=4
```

## Python entry points

Run the following commands from the parent directory of this package. Here, `submit` is the package name:

```sh
cd /home/wzx/code/mIFDec2/manual

# Process one core with a single process.
/home/wzx/miniconda3/envs/wsi/bin/python -m submit.run_sdf_wsi_single_worker \
    --dataset comet --task single --sample-id core01 \
    --save-dir /data/wzx/mIFDec/comet_compare/sdf \
    --sdf-params submit/sdf_params.json

# Process all cores with multiple worker processes.
/home/wzx/miniconda3/envs/wsi/bin/python -m submit.run_sdf_wsi_muti_workers \
    --dataset comet --task cohort --worker 4 \
    --save-dir /data/wzx/mIFDec/comet_compare/sdf \
    --sdf-params submit/sdf_params.json
```

When calling a Python entry point directly, omitting `--tile-size` selects 512 pixels for COMET and 256 pixels for the other datasets. `--sample-id` accepts multiple sample IDs. Multiprocessing runners support `--overwrite`; single-process runners overwrite files at matching output paths.

For other methods, replace `sdf` in the module name with the corresponding method identifier. LMSF accepts `--lmsf-params`, Sauvola accepts `--sauvola-params`, and BaSiC accepts `--basic-params` and `--fit-samples`. LMSF and Sauvola load the adjacent JSON files by default. Without `--sdf-params`, ROSBD uses the defaults in `method.py`. The COMET integration did not change any method parameters.

The default COMET input in this directory includes DAPI, and `sdf_params.json` currently specifies `q_b=0.4, q_a=0.4`. The manuscript's cellular experiment used eight marker channels without DAPI and `q_b=0.15, q_a=0.4`. To reproduce that experiment, explicitly select the corresponding input channels and parameters. RNA-derived labels and Nimbus cellular evaluation are handled by `compare/cellular`, rather than these image-processing runners.

## Output layout

```text
SAVE_DIR/sample_id/channel_name/
    raw/x_y.tif
    raw/x_y.png
    result/x_y.tif
    result/x_y.png
    bg/x_y.tif
    bg/x_y.png
    result_raw_combinded/x_y.png
```

TIFF files retain the original intensity scale. PNG files are display previews; side-by-side comparisons show the corrected image on the left and the raw image on the right. Preview generation uses the existing logic in `utils.py` and does not affect TIFF values.

## Environment

Use Python 3.10 and install the release dependencies from the root requirements file:

```sh
python -m pip install -r requirements.txt
```

`requirements.txt` covers image preprocessing, ROSBD, LMSF, Sauvola, BaSiC, CLEAR-AF, and the quantitative figure scripts. Standard-library modules and repository-local imports are not listed as pip packages. The `baseline/` directory contains the four method wrappers.

The numerical versions are selected to satisfy the published package constraints: BaSiCPy 2.0.0 requires SciPy below 1.13, so this file uses SciPy 1.12.0, NumPy 1.26.4, and OpenCV 4.11.0.86. It is a compatible installation specification, not an exact freeze of the existing server environment. The existing `wsi` environment has SciPy 1.15.3, which does not satisfy BaSiCPy's declared constraint; that environment has not been changed. A fresh environment is recommended for installing the complete release dependency set.

BaSiCPy 2.0.0 uses PyTorch. The `torch==2.6.0` requirement accepts a matching CPU or CUDA build; install the appropriate PyTorch build for your hardware before the remaining requirements if necessary. JAX is not required. CLEAR-AF is installed from the fixed upstream commit recorded in its wrapper, rather than from a server-local installation archive.

`openslide-bin` supplies the OpenSlide shared library on supported platforms. If no compatible binary is available for your operating system, install OpenSlide through the system package manager instead. Only `opencv-python-headless` is listed; avoid installing it alongside `opencv-python` in a new environment.

`data/patch_mIF.py` additionally imports the repository-local module `utils.patching_wsi`. Keep the repository root on `PYTHONPATH` when running that script; this local module cannot be installed through `requirements.txt`.
