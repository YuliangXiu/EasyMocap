# This is a simple demo of camera projection for body scanner.
# The radial distortion is ignored here.

# This code works with *.csd calibration files.
# It should work with captures done after around Jan 2021

import numpy as np

import os
from os.path import join

import matplotlib.pyplot as plt
import trimesh
import pyrender

from scipy.spatial.transform import Rotation


def run(scan_file, ref_img_file, cam_cali_file, camera_id, output_dir=''):
    
    os.makedirs(output_dir, exist_ok=True)

    def read_camera_cali(file, camera_id):
        with open(file) as f:
            lines = [line.rstrip() for line in f]

        camera_cali = {}

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
        t = np.array(camera_info[4:7]).astype(np.float32)/1000.0

        R = Rotation.from_rotvec(np.array([Rxyz[0], Rxyz[1], Rxyz[2]]))
        R = R.as_matrix()
        
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R
        T[:3, 3] = t
        T = np.linalg.inv(T)
        T_openGL = np.eye(4, dtype=np.float32)
        T_openGL[:3,:3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()

        T = np.dot(T, T_openGL)

        camera_cali['extrinsic'] = T
        camera_cali['fx'] = float(camera_info[7])
        camera_cali['fy'] = float(camera_info[8])
        camera_cali['c_x'] = float(camera_info[9])+0.5
        camera_cali['c_y'] = float(camera_info[10])+0.5
        
        return camera_cali

    scene = pyrender.Scene()

    # Read and add camera:
    camera_cali = read_camera_cali(cam_cali_file, camera_id)

    # read reference image:
    ref_img = plt.imread(ref_img_file)

    RENDER_RESOLUTION = [ref_img.shape[1], ref_img.shape[0]]
    UV = [camera_cali['c_x']*RENDER_RESOLUTION[0], camera_cali['c_y']*RENDER_RESOLUTION[1]]

    fx = RENDER_RESOLUTION[0]*camera_cali['fx']
    fy = RENDER_RESOLUTION[0]*camera_cali['fy']
    print('fx, fy\n', fx, fy, RENDER_RESOLUTION)

    camera = pyrender.camera.IntrinsicsCamera(fx, fy, UV[0], UV[1])
    # print('intrisics:\n', camera.get_projection_matrix(width=RENDER_RESOLUTION[0], height=RENDER_RESOLUTION[1]))
    camera_pose = camera_cali['extrinsic']
    print('extrinsic\n', camera_pose)

    scene.add(camera, pose=camera_pose)

    # Add light:
    light = pyrender.SpotLight(color=np.ones(3), intensity=50.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)

    # Add mesh:
    scan_mesh = trimesh.load(scan_file, process=False)
    scan_mesh.vertices /= 1000.0

    mesh = pyrender.Mesh.from_trimesh(scan_mesh)
    scene.add(mesh)

    # Render
    r = pyrender.OffscreenRenderer(RENDER_RESOLUTION[0], RENDER_RESOLUTION[1])
    color, depth = r.render(scene)
    output_image_file = os.path.basename(ref_img_file)[:-4]+'_rendering.png'
    # plt.imsave(join(output_dir, output_image_file), np.asarray(color), dpi = 1)

    # Post processing, overlay two images
    result_img = color.astype(np.float32)/255.0

    ref_img = ref_img.astype(np.float32)/255.0
    # result_img[:,:,0:2] *= 0
    result_img = 1.0 - ref_img + 1.0 - result_img
    result_img = 1.0 - result_img
    result_img = np.clip(result_img, 0.0, 1.0)

    # Save result:
    output_image_file = os.path.basename(ref_img_file)[:-4]+'_overlay.png'
    plt.imsave(join(output_dir, output_image_file), np.asarray(result_img), dpi = 1)

    return 

# for i in [6,7,8,17,18]:
for i in range(1,23,1):
    
    person_id = "00170"
    scan_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/scan.obj"
    ref_img_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/images/{i:02d}_C.jpg"
    cam_cali_file = f"data/PuzzleIOI/fitting/{person_id}/outfit5/camera.csd"

    print("Camera:", str(i).zfill(2))
    run(scan_file, ref_img_file, cam_cali_file, camera_id = str(i).zfill(2)+'_C', output_dir='tmp')
    print("______________________________________________")