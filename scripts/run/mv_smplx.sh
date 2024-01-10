export src_data="/ps/data/DynamicClothCap"
export tgt_data="./data/PuzzleIOI"

# 0. collect data from the clusters
python scripts/preprocess/copy_ioi_from_cluster.py \
    --src_path ${src_data} \
    --tgt_path ${tgt_data} \

# 1. extract skeletons
python scripts/preprocess/extract_skeletons.py \
    ${tgt_data}/fitting \

# # 2. example for SMPL-X reconstruction
# python apps/demo/mv1p.py ${tgt_data}/fitting \
#     --out ${data}/output/smplx \
#     --vis_det \
#     --vis_repro \
#     --undis \
#     --sub_vis 1 7 13 19 \
#     --body bodyhandface \
#     --model smplx \
#     --gender male \
#     --vis_smpl
