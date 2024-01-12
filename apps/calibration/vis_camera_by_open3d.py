'''
  @ Date: 2022-09-26 16:32:19
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-10-17 13:05:28
  @ FilePath: /EasyMocapPublic/apps/calibration/vis_camera_by_open3d.py
'''
import os

import cv2
import numpy as np
import open3d as o3d

# copy these functions explicitly to use them smoothly even without easymocap environment

Vector3dVector = o3d.utility.Vector3dVector


def create_pcd(xyz, color=None, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(xyz[:, :3])
    if color is not None:
        pcd.paint_uniform_color(color)
    if colors is not None:
        pcd.colors = Vector3dVector(colors)
    return pcd


def generate_colorbar(N=20, cmap='jet', rand=True, ret_float=False, ret_array=False, ret_rgb=False):
    bar = ((np.arange(N) / (N - 1)) * 255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    if rand:
        import random
        random.seed(666)
        index = [i for i in range(N)]
        random.shuffle(index)
        rgb = colorbar[index, :]
    else:
        rgb = colorbar
    if ret_rgb:
        rgb = rgb[:, ::-1]
    if ret_float:
        rgb = rgb / 255.
    if not ret_array:
        rgb = rgb.tolist()
    return rgb


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write(
                '  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)]))
            )
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        H = intri.read('H_{}'.format(cam), dt='int')
        W = intri.read('W_{}'.format(cam), dt='int')
        if H is None or W is None:
            print('[camera] no H or W for {}'.format(cam))
            H, W = -1, -1
        cams[cam]['H'] = H
        cams[cam]['W'] = W
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        assert Rvec is not None, cam
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = -Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
        if cams[cam]['dist'] is None:
            cams[cam]['dist'] = intri.read('D_{}'.format(cam))
            if cams[cam]['dist'] is None:
                print('[camera] no dist for {}'.format(cam))
    cams['basenames'] = cam_names
    return cams


def read_cameras(path, intri='intri.yml', extri='extri.yml', subs=[]):
    cameras = read_camera(os.path.join(path, intri), os.path.join(path, extri))
    cameras.pop('basenames')
    if len(subs) > 0:
        cameras = {key: cameras[key].astype(np.float32) for key in subs}
    return cameras


def transform_cameras(cameras):
    dims = {'x': 0, 'y': 1, 'z': 2}
    R_global = np.eye(3)
    T_global = np.zeros((3, 1))
    # order: trans0, rot, trans
    if len(args.trans0) == 3:
        trans = np.array(args.trans0).reshape(3, 1)
        T_global += trans
    if len(args.rot) > 0:
        for i in range(len(args.rot) // 2):
            dim = args.rot[2 * i]
            val = float(args.rot[2 * i + 1])
            rvec = np.zeros((3, ))
            rvec[dims[dim]] = np.deg2rad(val)
            R = cv2.Rodrigues(rvec)[0]
            R_global = R @ R_global
        T_global = R_global @ T_global
    # 平移相机
    if len(args.trans) == 3:
        trans = np.array(args.trans).reshape(3, 1)
        T_global += trans
    trans = np.eye(4)
    trans[:3, :3] = R_global
    trans[:3, 3:] = T_global

    # apply the transformation of each camera
    for key, cam in cameras.items():
        RT = np.eye(4)
        RT[:3, :3] = cam['R']
        RT[:3, 3:] = cam['T']
        RT = RT @ np.linalg.inv(trans)
        cam.pop('Rvec', '')
        cam['R'] = RT[:3, :3]
        cam['T'] = RT[:3, 3:]
    return cameras, trans


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--subs', type=str, default=[], nargs='+')
    parser.add_argument('--pcd', type=str, default=[], nargs='+')
    parser.add_argument('--trans0', type=float, nargs=3, default=[], help='translation')
    parser.add_argument('--rot', type=str, nargs='+', default=[], help='control the rotation')
    parser.add_argument('--trans', type=float, nargs=3, default=[], help='translation')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    body = None
    cameras = read_cameras(args.path)
    cameras, trans = transform_cameras(cameras)
    

    print(repr(trans))
    
    for pcd in args.pcd:
        if not os.path.exists(pcd):
            print(pcd, ' not exist')
            continue
        if pcd.endswith('.npy'):
            data = np.load(pcd)
            points = data[:, :3]
            colors = data[:, 3:]
            points = (trans[:3, :3] @ points.T + trans[:3, 3:]).T
            p = create_pcd(points, colors=data[:, 3:])
            body = p
        elif pcd.endswith('.ply'):
            print('Load pcd: ', pcd)
            p = o3d.io.read_point_cloud(pcd)
            print(p)
            body = p
        elif pcd.endswith('.pkl'):
            p = o3d.io.read_triangle_mesh(pcd)
            body = p
        elif pcd.endswith('.obj'):
            p = o3d.io.read_triangle_mesh(pcd)
            vertices = np.asarray(p.vertices) / 1000.0
            vertices = vertices[:, [0, 2, 1]]
            vertices[:,1] *= -1
            p.vertices = Vector3dVector(vertices)
            print(vertices.shape)
            print(vertices.min(axis=0))
            print(vertices.max(axis=0))
            body = p

    # floor 
    floor_width = 4.0
    floor_height = 4.0
    floor = o3d.geometry.TriangleMesh.create_box(
        width=floor_width, height=floor_height, depth=0.02
    )
    floor.paint_uniform_color([0, 0, 0])
    floor.translate([-floor_width/2.0, -floor_height/2.0, -0.6])
    
    # center axis
    center_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    
    app = gui.Application.instance
    app.initialize()
    w = app.create_window("Camera Vis", 2048, 2048)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    w.add_child(widget3d)
    widget3d.scene.add_geometry('floor', floor, mat)
    widget3d.scene.add_geometry('body', body, mat)
    widget3d.scene.add_geometry('center_axis', center_axis, mat)
    
    for ic, (cam, camera) in enumerate(cameras.items()):
        if len(args.subs) > 0 and cam not in args.subs:
            continue
        center = -camera['R'].T @ camera['T']
        print('{}: {}'.format(cam, center.T[0]))
        # if cam == '07_C':
        #     continue
        cam_coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[center[0, 0], center[1, 0], center[2, 0]]
        )
        cam_coords.rotate(camera['R'].T)
        
        # TODO: add label
        widget3d.add_3d_label(center[:,0], cam)
        widget3d.scene.add_geometry(cam, cam_coords, mat)
        
    app.run()
        