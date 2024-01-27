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

# multi-thread
from functools import partial
from glob import glob
from multiprocessing import Pool
from os.path import join
from termcolor import colored

import cv2
import matplotlib.pyplot as plt
import pyrender
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

mkdir = lambda x: os.makedirs(x, exist_ok=True)

import json

cameras = {}

person_id = "00145"

cam_cali_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"

motion_num = 10
camera_num = 10


def read_camera_cali(file, ref_img_file, camera_id):

    with open(file) as f:
        lines = [line.rstrip() for line in f]

    camera_cali = {}

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]

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

    camera_cali['c_x'] *= RENDER_RESOLUTION[0]
    camera_cali['c_y'] *= RENDER_RESOLUTION[1]
    camera_cali['fx'] *= RENDER_RESOLUTION[0]
    camera_cali['fy'] *= RENDER_RESOLUTION[0]

    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )

    camera_cali['intrinsic'] = camera.get_projection_matrix(
        width=RENDER_RESOLUTION[0], height=RENDER_RESOLUTION[1]
    )
    return camera_cali

scene = pyrender.Scene()
light = pyrender.SpotLight(
    color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
)

def render(scan_file, ref_img_file, camera_cali):

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]
    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )
    camera_pose = camera_cali['extrinsic']
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh = trimesh.intersections.slice_mesh_plane(scan_mesh, [0, 1, 0], [0, -585.0, 0])
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(RENDER_RESOLUTION[0], RENDER_RESOLUTION[1])
    color, _ = r.render(scene)
    mask = (color == color[0, 0]).sum(axis=2, keepdims=True) != 3
    masked_img = ref_img * mask
    
    scene.clear()

    return masked_img


for i in range(1, 23, 1):

    camera_id = f"{i:02d}_C"
    ref_img_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/images/{camera_id}.jpg"
    cameras[camera_id] = read_camera_cali(cam_cali_file, ref_img_file, camera_id)


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
    front_cam = len(glob(join(path, "*07_C.jpg"))) >= 1

    if len(src_meshs) == 1 and full_cam and front_cam:
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

    if len(outfit_seq_valid) > motion_num:

        outfit_seq_random = random.sample(outfit_seq_valid, motion_num)
        outfit_seq_front = random.sample([
            item for item in outfit_seq_valid if item not in outfit_seq_random
        ], min(len(outfit_seq_valid) - motion_num, 2*motion_num))

        # copy random views
        for outfit_id, outfit_seq in enumerate(outfit_seq_random):

            puzzle_captures = random.sample(glob(join(outfit_seq, "*_C.jpg")), camera_num)
            
            # copy front view
            for outfit_id, outfit_seq in enumerate(outfit_seq_front):
                front_path = glob(join(outfit_seq, "*07_C.jpg"))[0]
                out_path = join(
                    tgt_dir, f"{outfit_id+1+(motion_num*camera_num):03d}.jpg"
                )
                if (not os.path.exists(out_path)) or (overwrite):
                    outfit_seq_name = ".".join(os.path.basename(front_path).split(".")[:2])
                    scan_file = f"{os.path.dirname(front_path)}/../../meshes/{outfit_seq_name}.obj"
                    masked_img = render(scan_file, front_path, cameras["07_C"])
                    plt.imsave(out_path, np.asarray(masked_img), dpi=1)

            for cam_id, puzzle_capture in enumerate(puzzle_captures):
                out_path = join(tgt_dir, f"{outfit_id*camera_num+cam_id+1:03d}.jpg")
                masked_img = None
                if (not os.path.exists(out_path)) or (overwrite):
                    outfit_seq_name = ".".join(os.path.basename(puzzle_capture).split(".")[:2])
                    cam_name = os.path.basename(puzzle_capture).split(".")[-2]
                    scan_file = f"{os.path.dirname(puzzle_capture)}/../../meshes/{outfit_seq_name}.obj"
                    if os.path.exists(scan_file):
                        masked_img = render(scan_file, puzzle_capture, cameras[cam_name])
                    else:
                        print(colored(f"Missing {scan_file}", "red"))
                    plt.imsave(out_path, np.asarray(masked_img), dpi=1)


    else:
        print(colored(f"Not enough motions for {subject_dir} outfit {outfit_name}", "red"))
        if os.path.exists(f"{tgt_dir}/.."):
            shutil.rmtree(f"{tgt_dir}/..")

    print(f"Finish copying {tgt_dir}")


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

        outfit_names_tqdm = tqdm(outfit_names)
        for outfit_name in outfit_names_tqdm:
            outfit_names_tqdm.set_description(f"Copying {person_id} : {outfit_name}")
            tgt_outfit_dir = join(tgt_path, "puzzle", f"{person_id}/{outfit_name}/image")
            mkdir(tgt_outfit_dir)
            # remove the old files
            for file in glob(join(tgt_outfit_dir, "*.jpg")):
                if len(file.split("/")[-1].split(".")[-2])==2:
                    os.remove(file)
            copy_all(subject_dir, outfit_name, tgt_outfit_dir, overwrite)

    print(colored(f"Finish copying {subject}", "green"))


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

    print("CPU:", mp.cpu_count())
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
