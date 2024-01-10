'''
  @ Date: 2021-01-08 15:05:05
  @ Author: Yuliang Xiu
  @ LastEditors: Yuliang Xiu
  @ LastEditTime: 2021-01-08 15:05:05
  @ FilePath: /EasyMocap/scripts/preprocess/copy_ioi_from_cluster.py
'''

import os
import fnmatch
import shutil
from glob import glob
from os.path import join
from torchvision.utils import make_grid
from torchvision.io import read_image

# multi-thread
from functools import partial
from multiprocessing import Pool, Queue
import multiprocessing as mp

import cv2
from tqdm import tqdm
from termcolor import colored

from easymocap.mytools.debug_utils import myerror, mywarn

mkdir = lambda x: os.makedirs(x, exist_ok=True)

import json


def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def copy_all(src_dir, tgt_dir):

    src_meshs = glob(join(src_dir, "meshes", "*000001.obj"))

    if len(src_meshs) == 1 and (not os.path.exists(join(tgt_dir, "scan.obj"))):

        for type in ["images"]:
            mkdir(join(tgt_dir, type))
        for src_img in glob(join(src_dir, "images/000001", "*.jpg")):
            tgt_img = join(tgt_dir, "images", os.path.basename(src_img)[-8:])
            if not os.path.exists(tgt_img):
                shutil.copyfile(src_img, tgt_img)

        src_meshs = glob(join(src_dir, "meshes", "*000001.obj"))

        src_mesh = src_meshs[0]

        src_cam = glob(join(src_dir, "meshes", "*.csd"))[0]
        tgt_mesh = join(tgt_dir, "scan.obj")
        tgt_cam = join(tgt_dir, "camera.csd")

        shutil.copyfile(src_mesh, tgt_mesh)
        shutil.copyfile(src_cam, tgt_cam)

def copy_dataset(subject, src_path, tgt_path):

    person_id = subject.split("_")[-2]
    motion_dir = join(src_path, subject)

    if os.path.isdir(motion_dir):

        mkdir(join(tgt_path, "fitting", person_id))
        motions = sorted(os.listdir(motion_dir))
        apose_names = fnmatch.filter(motions, "*minimal_A[Pp]ose")
        outfit_seq_names = fnmatch.filter(motions, "*outfit*_seq[0-9]*")

        if len(apose_names) == 1:

            src_apose_dir = join(motion_dir, apose_names[0])
            tgt_apose_dir = join(tgt_path, "fitting", person_id, "apose")
            copy_all(src_apose_dir, tgt_apose_dir)

        else:
            mywarn(f"no apose in {motion_dir}")

        for outfit_name in sorted(outfit_seq_names):
            tgt_outfit_name = [
                outfit_name for outfit_name in outfit_name.split("_") if "outfit" in outfit_name
            ][0]
            src_outfit_dir = join(motion_dir, outfit_name)
            tgt_outfit_dir = join(tgt_path, "fitting", person_id, tgt_outfit_name)
            
            copy_all(src_outfit_dir, tgt_outfit_dir)

        all_front_paths = sorted(glob(join(tgt_path, "fitting", person_id, "*/images/07_C.jpg")))
        
        try:
            all_full_imgs = [
                read_image(img_path)
                for img_path in all_front_paths
            ]
            cv2.imwrite(
                join(tgt_path, "fitting", person_id, "full.jpg"),
                make_grid(all_full_imgs, nrow=5).permute(1, 2, 0).numpy()[:, :, ::-1]
            )
        except:
            myerror(f"cannot read {all_front_paths}")    

    # print(f"{person_id} is done!")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    args = parser.parse_args()

    mkdir(join(args.tgt_path, "fitting"))
    subjects = sorted(os.listdir(args.src_path))
    
    # subjects = ["DynamicClothCap_220426_03612_CG"]

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        pool.map(partial(copy_dataset, src_path=args.src_path, tgt_path=args.tgt_path), subjects)
        pool.close()
        pool.join()
