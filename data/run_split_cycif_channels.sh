# Edit these three values to change the job inputs.
mIF_dir=./datasets/wala-christensen-2026-pik3ca-crc/cycif
output_dir=./datasets/wala-christensen-2026-pik3ca-crc/cycif_sp
n_threads=4

channel_names=./datasets/wala-christensen-2026-pik3ca-crc/cycif_channels.txt
PYTHON=${PYTHON:-python}
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
# Resolve input and output paths from the release directory.
cd "$script_dir/.."

exec "$PYTHON" "$script_dir/split_cycif_channels.py" \
    --mif-dir "$mIF_dir" \
    --output-dir "$output_dir" \
    --channel-names "$channel_names" \
    --n-threads "$n_threads" \
    "$@"
