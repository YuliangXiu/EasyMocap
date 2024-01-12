# This is a simple demo of camera projection for body scanner.
# The radial distortion is ignored here.

# This code works with *.csd calibration files.
# It should work with captures done after around Jan 2021

import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from easymocap.mytools.camera_utils import FileStorage


def write_extri(extri_name, cameras):

    extri = FileStorage(extri_name, True)
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')

    for key_, val in cameras.items():
        key = key_.split('.')[0]

        RT_openGL = np.eye(4, dtype=np.float32)
        RT_openGL[:3, :3] = Rotation.from_rotvec(np.pi * np.array([0.5, 0, 0])).as_matrix()
        RT = np.dot(RT_openGL, val['extrinsic'])
        RT[:3, :3] = np.dot(
            RT[:3, :3],
            Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        )

        RT = np.linalg.inv(RT)
        R = RT[:3, :3]
        T = RT[:3, 3].reshape(3, 1)

        R_vec, _ = cv2.Rodrigues(R)
        extri.write('R_{}'.format(key), R_vec)
        extri.write('Rot_{}'.format(key), R)
        extri.write('T_{}'.format(key), T)

    return 0


def write_intri(extri_name, cameras):

    extri = FileStorage(extri_name, True)
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')

    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K = np.array([[val['fx'], 0., val['c_x']], [0., val['fy'], val['c_y']], [0., 0., 1.]])
        # K = val['intrinsic'][:3, :3]
        dist = np.array([0., 0., 0., 0., 0.]).reshape(5, 1)
        extri.write('K_{}'.format(key), K)
        extri.write('dist_{}'.format(key), dist)

    return 0


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


def render(scan_file, ref_img_file, camera_cali, output_dir=''):

    os.makedirs(output_dir, exist_ok=True)

    scene = pyrender.Scene()

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]
    camera = pyrender.camera.IntrinsicsCamera(
        camera_cali['fx'], camera_cali['fy'], camera_cali['c_x'], camera_cali['c_y']
    )
    print(
        'intrisics:\n',
        camera.get_projection_matrix(width=RENDER_RESOLUTION[0], height=RENDER_RESOLUTION[1])
    )
    camera_pose = camera_cali['extrinsic']
    print('extrinsic\n', camera_pose)
    print(RENDER_RESOLUTION)
    # import ipdb; ipdb.set_trace()

    scene.add(camera, pose=camera_pose)

    # Add light:
    light = pyrender.SpotLight(
        color=np.ones(3), intensity=50.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0
    )
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(RENDER_RESOLUTION[0], RENDER_RESOLUTION[1])
    color, depth = r.render(scene)
    output_image_file = os.path.basename(ref_img_file)[:-4] + '_rendering.png'
    # plt.imsave(join(output_dir, output_image_file), np.asarray(color), dpi = 1)

    # Post processing, overlay two images
    result_img = color.astype(np.float32) / 255.0

    ref_img = ref_img.astype(np.float32) / 255.0
    # result_img[:,:,0:2] *= 0
    result_img = 1.0 - ref_img + 1.0 - result_img
    result_img = 1.0 - result_img
    result_img = np.clip(result_img, 0.0, 1.0)

    # Save result:
    output_image_file = os.path.basename(ref_img_file)[:-4] + '_overlay.png'
    plt.imsave(join(output_dir, output_image_file), np.asarray(result_img), dpi=1)


if __name__ == '__main__':

    person_id = "00170"
    scan_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/scan.obj"
    cam_cali_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"

    # exported files
    intri_file = f"data/PuzzleIOI/intri.yml"
    extri_file = f"data/PuzzleIOI/extri.yml"

    cameras = {}

    for i in range(1, 23, 1):

        camera_id = f"{i:02d}_C"
        ref_img_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/images/{camera_id}.jpg"
        cameras[camera_id] = read_camera_cali(cam_cali_file, ref_img_file, camera_id)
        # render(scan_file, ref_img_file, cameras[camera_id], output_dir='tmp')

    write_extri(extri_file, cameras)
    write_intri(intri_file, cameras)

    # import pprint
    # pprint.pprint(cameras)
