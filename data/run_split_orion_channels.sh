# Edit these three values to change the job inputs.
mIF_dir=./datasets/wala-christensen-2026-pik3ca-crc/orion
output_dir=./datasets/wala-christensen-2026-pik3ca-crc/orion_sp
n_threads=4

PYTHON=${PYTHON:-python}
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
# Resolve input and output paths from the release directory.
cd "$script_dir/.."

exec "$PYTHON" "$script_dir/split_orion_channels.py" \
    --mif-dir "$mIF_dir" \
    --output-dir "$output_dir" \
    --n-threads "$n_threads" \
    "$@"
