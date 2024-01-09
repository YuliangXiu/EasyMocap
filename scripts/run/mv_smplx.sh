export data="./data/zju"

# 0. extract the video to images
# python3 scripts/preprocess/extract_video.py ${data} --handface

# 2.2 example for SMPL-X reconstruction
python apps/demo/mv1p.py ${data} --out ${data}/output/smplx --vis_det --vis_repro --undis --sub_vis 1 7 13 19 --body bodyhandface --model smplx --gender male --vis_smpl
