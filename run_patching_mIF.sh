PATCHSIZE=256
ROOT=/data/wzx/BM-Her2N75-15
python data/patch_mIF.py \
    --mif-dir $ROOT \
    --save-dir $ROOT/patch_info/"$PATCHSIZE"_"$PATCHSIZE" \
    --patch-size $PATCHSIZE \
    --overlap 0 \
    --n-threads 32 \
    --stitch