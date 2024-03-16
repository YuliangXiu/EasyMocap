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
from tqdm import tqdm
from shutil import copyfile

# multi-thread
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join
from termcolor import colored

import matplotlib.pyplot as plt
import pyrender
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

mkdir = lambda x: os.makedirs(x, exist_ok=True)

import json

cameras = np.load("./scripts/preprocess/camera.npy", allow_pickle=True).item()

motion_num = 10
camera_num = 10


def read_camera_cali(file, ref_img_file, camera_id):

    with open(file) as f:
        lines = [line.rstrip() for line in f]

    camera_cali = {}

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    line_id = None
    for i in range(len(lines)):
        if lines[i].split()[0] == camera_id:
            line_id = i
            break
    if line_id is None:
        print("Wrong camera id!")
        exit()

    camera_info = lines[line_id].split()

    Rxyz = np.array(camera_info[1:4]).astype(np.float32)
    t = np.array(camera_info[4:7]).astype(np.float32) / 1000.0

    R = Rotation.from_rotvec(np.array([Rxyz[0], Rxyz[1], Rxyz[2]]))
    R = R.as_matrix()

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    T = np.linalg.inv(T)
    T_openGL = np.eye(4, dtype=np.float32)
    T_openGL[:3, :3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

    T = np.dot(T, T_openGL)
    camera_cali['extrinsic'] = T

    # intrinsics camera
    camera_cali['fx'] = float(camera_info[7])
    camera_cali['fy'] = float(camera_info[8])
    camera_cali['c_x'] = float(camera_info[9]) + 0.5
    camera_cali['c_y'] = float(camera_info[10]) + 0.5

    camera_cali['c_x'] *= ref_img.shape[1]
    camera_cali['c_y'] *= ref_img.shape[0]
    camera_cali['fx'] *= ref_img.shape[1]
    camera_cali['fy'] *= ref_img.shape[1]

    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )

    camera_cali['intrinsic'] = camera.get_projection_matrix(
        width=ref_img.shape[1], height=ref_img.shape[0]
    )
    return camera_cali


scene = pyrender.Scene()
light = pyrender.SpotLight(
    color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
)


def render(scan_file, ref_img_file, camera_cali):

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )
    camera_pose = camera_cali['extrinsic']
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh = trimesh.intersections.slice_mesh_plane(scan_mesh, [0, 1, 0], [0, -580.0, 0])
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(ref_img.shape[1], ref_img.shape[0])
    color, _ = r.render(scene)
    mask = (color == color[0, 0]).sum(axis=2, keepdims=True) != 3
    masked_img = ref_img * mask

    scene.clear()

    return masked_img


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
    fname = path.split("/")[-2]
    has_mesh = len(glob(join(path, "../../meshes", f"*{fname}.obj"))) >= 1
    enough_cam = len(glob(join(path, "*_C.jpg"))) >= camera_num
    front_cam = len(glob(join(path, "*07_C.jpg"))) >= 1

    if has_mesh and enough_cam and front_cam:
        return True
    else:
        return False


def copy_all(outfit_name, subject_dir, tgt_dir):

    tgt_dir = join(tgt_dir, outfit_name, "image")
    mkdir(tgt_dir)

    outfit_seq_dirs = sorted(
        glob(join(subject_dir, f"*{outfit_name}_seq*/images/00*/")), reverse=True
    )
    outfit_seq_valid = [path for path in outfit_seq_dirs if check_img_valid(path)]

    if len(outfit_seq_valid) > motion_num:

        outfit_seq_random = random.sample(outfit_seq_valid, motion_num)
        outfit_seq_front = random.sample([
            item for item in outfit_seq_valid if item not in outfit_seq_random
        ], min(len(outfit_seq_valid) - motion_num, 2 * motion_num))

        # copy front view
        for outfit_id, outfit_seq in enumerate(outfit_seq_front):

            front_path = glob(join(outfit_seq, "*07_C.jpg"))[0]
            out_path = join(tgt_dir, f"{outfit_id+1+(motion_num*camera_num):03d}_07_C.jpg")
            raw_path = join(tgt_dir, f"{outfit_id+1+(motion_num*camera_num):03d}_07_C_raw.jpg")

            if (not os.path.exists(out_path)):
                outfit_seq_name = ".".join(os.path.basename(front_path).split(".")[:2])
                scan_file = f"{os.path.dirname(front_path)}/../../meshes/{outfit_seq_name}.obj"
                if os.path.exists(scan_file):
                    masked_img = render(scan_file, front_path, cameras["07_C"])
                    plt.imsave(out_path, np.asarray(masked_img), dpi=1)
                    copyfile(front_path, raw_path)
                else:
                    print(colored(f"Cannot find {scan_file}", "red"))
                    continue

        # copy random views
        for outfit_id, outfit_seq in enumerate(outfit_seq_random):

            puzzle_captures = random.sample(glob(join(outfit_seq, "*_C.jpg")), camera_num)

            for cam_id, puzzle_capture in enumerate(puzzle_captures):

                cam_name = os.path.basename(puzzle_capture).split(".")[-2]
                out_path = join(tgt_dir, f"{outfit_id*camera_num+cam_id+1:03d}_{cam_name}.jpg")
                same_ids_out_paths = glob(f"{tgt_dir}/{outfit_id*camera_num+cam_id+1:03d}_*.jpg")

                if len(same_ids_out_paths) < 1:

                    outfit_seq_name = ".".join(os.path.basename(puzzle_capture).split(".")[:2])
                    cam_name = os.path.basename(puzzle_capture).split(".")[-2]
                    scan_file = f"{os.path.dirname(puzzle_capture)}/../../meshes/{outfit_seq_name}.obj"

                    if os.path.exists(scan_file):
                        masked_img = render(scan_file, puzzle_capture, cameras[cam_name])
                        plt.imsave(out_path, np.asarray(masked_img), dpi=1)
                    else:
                        print(colored(f"Cannot find {scan_file}", "red"))
                        continue
                    
        print(colored(f"Finish copying {subject_dir} outfit {outfit_name}", "green"))
    else:
        print(colored(f"Not enough motions for {subject_dir} outfit {outfit_name}", "red"))

        if os.path.exists(os.path.dirname(tgt_dir)):
            shutil.rmtree(os.path.dirname(tgt_dir))
            
    


def copy_dataset(subject, src_path, tgt_path):

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

        print("CPU:", mp.cpu_count())
        print("propress", len(outfit_names))

        with Pool(processes=len(outfit_names), maxtasksperchild=1) as pool:
            pool.map(
                partial(
                    copy_all,
                    subject_dir=subject_dir,
                    tgt_dir=join(tgt_path, "puzzle_cam", person_id),
                ), outfit_names
            )
            pool.close()
            pool.join()

    print(colored(f"Finish copying {subject}", "green"))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str)
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--tgt_path', type=str)
    args = parser.parse_args()

    mkdir(join(args.tgt_path, "puzzle_cam"))

    copy_dataset(
        subject=args.subject,
        src_path=args.src_path,
        tgt_path=args.tgt_path,
    )
