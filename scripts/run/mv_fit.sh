#!/bin/bash
getenv=True
source /home/yxiu/miniconda3/bin/activate TeCH

export HF_HOME="/is/cluster/yxiu/.cache"
export tgt_data="./data/PuzzleIOI"
export PYOPENGL_PLATFORM="egl"

# 2. example for SMPL-X reconstruction
python apps/demo/mv1p_ioi.py ${tgt_data}/fitting/$1 \
    --out ${tgt_data}/fitting/$1/output/smplx \
    --annot skeletons \
    --vis_det \
    --vis_repro \
    --undis \
    --gender $2 \
    --start 0 \
    --end 1 \
    --sub_vis 03_C 07_C 11_C 15_C 19_C 22_C \
    --body body15 \
    --model smplx \
    --vis_smpl