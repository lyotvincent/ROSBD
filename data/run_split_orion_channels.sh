#!/bin/sh
set -eu

# Edit these three values to change the job inputs.
mIF_dir=/data/wzx/wala-christensen-2026-pik3ca-crc/orion
output_dir=/data/wzx/wala-christensen-2026-pik3ca-crc/orion_sp
n_threads=4

conda=/home/wzx/miniconda3/bin/conda
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

exec "$conda" run --no-capture-output -n wsi \
    python "$script_dir/split_orion_channels.py" \
    --mif-dir "$mIF_dir" \
    --output-dir "$output_dir" \
    --n-threads "$n_threads" \
    "$@"
