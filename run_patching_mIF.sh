PATCHSIZE=256
ROOT=./datasets/BM-Her2N75-15
PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
"$PYTHON" data/patch_mIF.py \
    --mif-dir "$ROOT" \
    --save-dir "$ROOT/patch_info/${PATCHSIZE}_${PATCHSIZE}" \
    --patch-size $PATCHSIZE \
    --overlap 0 \
    --n-threads 32 \
    --stitch
