# Edit the settings below, then run: sh run_method.sh
METHOD=clearaf                 # sdf / clearaf / lmsf / sauvola / basic
TASK=cohort                # single / cohort
DATASET=orion              # orion / cycif / bmcycif / comet
SAMPLE_ID="BM-Her2N75-15-Scene-059"  # Ignored in cohort mode
TILE_SIZE=512              # COMET coordinates use 512-pixel tiles; match the existing tile size for other datasets
WORKER=16
OVERWRITE=${OVERWRITE:-False} # False: skip existing sample directories; True: reprocess
SAVE_DIR="/data/wzx/mIFDec/${DATASET}_compare/${METHOD}"
FIT_SAMPLES=128             # BaSiC only: maximum fitting inputs (complete views for BM, tiles otherwise)
CONDA_ROOT=/home/wzx/miniconda3

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PACKAGE_NAME=${SCRIPT_DIR##*/}
cd "$SCRIPT_DIR/.."
case "$DATASET" in
    orion|cycif|bmcycif|comet) ;;
    *) echo "Unknown DATASET: $DATASET" >&2; exit 1 ;;
esac
case "$METHOD" in
    sdf|clearaf|lmsf|sauvola|basic) ;;
    *) echo "Unknown METHOD: $METHOD" >&2; exit 1 ;;
esac

if [ "$METHOD" = clearaf ]; then
    case "$DATASET" in
        bmcycif|comet)
            echo "$DATASET has no AF reference configured; ClearAF is unavailable. Choose sdf / lmsf / sauvola / basic." >&2
            exit 1 ;;
    esac
fi

set -- -m "${PACKAGE_NAME}.run_${METHOD}_wsi_muti_workers" \
    --task "$TASK" --dataset "$DATASET" --tile-size "$TILE_SIZE" \
    --save-dir "$SAVE_DIR" --worker "$WORKER"

case "$OVERWRITE" in
    True|true|TRUE) set -- "$@" --overwrite ;;
    False|false|FALSE) ;;
    *) echo "OVERWRITE must be True or False: $OVERWRITE" >&2; exit 1 ;;
esac

case "$TASK" in
    single) : "${SAMPLE_ID:?SAMPLE_ID is required in single mode}"; set -- "$@" --sample-id "$SAMPLE_ID" ;;
    cohort) ;;
    *) echo "Unknown TASK: $TASK" >&2; exit 1 ;;
esac
# Automatically load the parameter JSON for the selected method.
case "$METHOD" in
    sdf|lmsf|sauvola)
        set -- "$@" "--${METHOD}-params" "${SCRIPT_DIR}/${METHOD}_params.json" ;;
esac
if [ "$METHOD" = basic ]; then
    set -- "$@" --fit-samples "$FIT_SAMPLES"
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$CONDA_ROOT/envs/wsi/bin/python" "$@"
