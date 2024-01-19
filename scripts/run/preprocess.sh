export src_data="/ps/data/DynamicClothCap"
export tgt_data="./data/PuzzleIOI"
export job_path="./scripts/run/jobs.txt"

# # 0. collect data from the clusters (for EasyMocap)
# python scripts/preprocess/copy_ioi_from_cluster.py \
#     --src_path ${src_data} \
#     --tgt_path ${tgt_data} \

# # 1. extract skeletons
# python scripts/preprocess/extract_skeletons.py \
#     ${tgt_data}/fitting \

# # 2. extract intri+extri cameras
# python scripts/preprocess/extract_camera.py

# # 3. generate jobs
# python scripts/preprocess/generate_jobs.py \
#     --tgt_path ${tgt_data}/fitting \
#     --job_path ${job_path} \

# 4. collect data from the clusters (for PuzzleAvatar)
python scripts/preprocess/copy_ioi_for_puzzleavatar.py \
    --src_path ${src_data} \
    --tgt_path ${tgt_data} \
##    --overwrite
