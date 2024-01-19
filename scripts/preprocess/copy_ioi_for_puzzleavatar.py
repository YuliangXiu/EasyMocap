'''
  @ Date: 2021-01-08 15:05:05
  @ Author: Yuliang Xiu
  @ LastEditors: Yuliang Xiu
  @ LastEditTime: 2021-01-08 15:05:05
  @ FilePath: /EasyMocap/scripts/preprocess/copy_ioi_from_cluster.py
'''

import fnmatch
import multiprocessing as mp
import os
import random
import shutil

# multi-thread
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join
from termcolor import colored

mkdir = lambda x: os.makedirs(x, exist_ok=True)

import json

motion_num = 10
camera_num = 10


def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def check_img_valid(path):
    src_meshs = glob(join(path, "../../meshes", "*000001.obj"))
    full_cam = len(glob(join(path, "*_C.jpg"))) >= camera_num

    if len(src_meshs) == 1 and full_cam:
        return True
    else:
        return False


def copy_all(subject_dir, outfit_name, tgt_dir, overwrite):

    outfit_seq_dirs = sorted(
        glob(join(subject_dir, f"*{outfit_name}_seq*/images/00*/")), reverse=True
    )
    outfit_seq_valid = [path for path in outfit_seq_dirs if check_img_valid(path)]
    # outfit_seq_invalid = [path for path in outfit_seq_dirs if not check_img_valid(path)]

    # if len(outfit_seq_invalid) >0:
    #     print(colored(f"Ignored lists: {outfit_seq_invalid}", "yellow"))

    if len(outfit_seq_valid) >= motion_num:

        outfit_seq_random = random.sample(outfit_seq_valid, motion_num)
        outfit_seq_front = random.sample([
            item for item in outfit_seq_valid if item not in outfit_seq_random
        ], motion_num)

        # copy random views
        for outfit_id, outfit_seq in enumerate(outfit_seq_random):

            puzzle_captures = random.sample(glob(join(outfit_seq, "*_C.jpg")), camera_num)

            for cam_id, puzzle_capture in enumerate(puzzle_captures):
                mkdir(tgt_dir)
                out_path = join(tgt_dir, f"{outfit_id*camera_num+cam_id+1:02d}.jpg")
                if (not os.path.exists(out_path)) or (overwrite):
                    shutil.copyfile(puzzle_capture, out_path)

        # copy front view
        for _, outfit_seq in enumerate(outfit_seq_front):
            front_path = glob(join(outfit_seq, "*07_C.jpg"))[0]
            out_path = join(tgt_dir, "00.jpg")
            if (not os.path.exists(out_path)) or (overwrite):
                shutil.copyfile(front_path, out_path)
    else:
        print(colored(f"Not enough motions for {subject_dir} outfit {outfit_name}", "red"))
        if os.path.exists(f"{tgt_dir}/.."):
            shutil.rmtree(f"{tgt_dir}/..")


def copy_dataset(subject, src_path, tgt_path, overwrite):

    person_id = subject.split("_")[-2]
    subject_dir = join(src_path, subject)

    if os.path.isdir(subject_dir):

        motions = sorted(os.listdir(subject_dir))
        outfit_seq_names_all = fnmatch.filter(motions, "*outfit*_seq[0-9]*")
        outfit_names = sorted(
            list(
                set([[item for item in outfit_name.split("_") if 'outfit' in item][0]
                     for outfit_name in outfit_seq_names_all])
            )
        )

        for outfit_name in outfit_names:
            tgt_outfit_dir = join(tgt_path, "puzzle", f"{person_id}/{outfit_name}/image")
            copy_all(subject_dir, outfit_name, tgt_outfit_dir, overwrite)

    print("Finish copying", subject)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    mkdir(join(args.tgt_path, "puzzle"))
    subjects = sorted(os.listdir(args.src_path))

    # subjects = ["DynamicClothCap_220426_03612_CG"]

    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        pool.map(
            partial(
                copy_dataset,
                src_path=args.src_path,
                tgt_path=args.tgt_path,
                overwrite=args.overwrite
            ), subjects
        )
        pool.close()
        pool.join()
