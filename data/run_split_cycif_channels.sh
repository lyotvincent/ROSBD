#!/bin/sh
set -eu

# Edit these three values to change the job inputs.
mIF_dir=/data/wzx/wala-christensen-2026-pik3ca-crc/cycif
output_dir=/data/wzx/wala-christensen-2026-pik3ca-crc/cycif_sp
n_threads=4

channel_names=/data/wzx/wala-christensen-2026-pik3ca-crc/cycif_channels.txt
conda=/home/wzx/miniconda3/bin/conda
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

exec "$conda" run --no-capture-output -n wsi \
    python "$script_dir/split_cycif_channels.py" \
    --mif-dir "$mIF_dir" \
    --output-dir "$output_dir" \
    --channel-names "$channel_names" \
    --n-threads "$n_threads" \
    "$@"
