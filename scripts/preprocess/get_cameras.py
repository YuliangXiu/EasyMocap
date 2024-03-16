import matplotlib.pyplot as plt
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation


def read_camera_cali(file, ref_img_file, camera_id):

    with open(file) as f:
        lines = [line.rstrip() for line in f]

    camera_cali = {}
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


person_id = "00145"
outfit_id="outfit5"
cam_cali_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/{outfit_id}/camera.csd"
cameras = {}

for i in range(1, 23, 1):

    camera_id = f"{i:02d}_C"
    ref_img_file = f"/ps/scratch/ps_shared/yxiu/PuzzleIOI/fitting/{person_id}/{outfit_id}/images/{camera_id}.jpg"
    cameras[camera_id] = read_camera_cali(cam_cali_file, ref_img_file, camera_id)

np.save("./scripts/preprocess/camera.npy", cameras)
