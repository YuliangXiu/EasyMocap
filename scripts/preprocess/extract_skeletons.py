'''
  @ Date: 2021-01-08 15:05:05
  @ Author: Yuliang Xiu
  @ LastEditors: Yuliang Xiu
  @ LastEditTime: 2021-01-08 15:05:05
  @ FilePath: /EasyMocap/scripts/preprocess/extract_skeletons.py
'''

import os
from glob import glob
from os.path import join

import shutil
import cv2
import numpy as np
from tqdm import tqdm


# multi-thread
from functools import partial
from multiprocessing import Pool, Queue
import multiprocessing as mp

mkdir = lambda x: os.makedirs(x, exist_ok=True)

import json


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def save_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)



config = {
    'openpose': {'root': '', 'res': 1, 'hand': False, 'face': False, 'vis': False, 'ext': '.jpg'},
    'openposecrop': {},
    'feet': {'root': '', 'res': 1, 'hand': False, 'face': False, 'vis': False, 'ext': '.jpg'},
    'feetcrop': {'root': '', 'res': 1, 'hand': False, 'face': False, 'vis': False, 'ext': '.jpg'},
    'yolo': {
        'ckpt_path': 'data/models/yolov4.weights',
        'conf_thres': 0.3,
        'box_nms_thres': 0.5,    # means keeping the bboxes that IOU<0.5
        'ext': '.jpg',
        'isWild': False,
    },
    'hrnet':
    {'nof_joints': 17, 'c': 48, 'checkpoint_path': 'data/models/pose_hrnet_w48_384x288.pth'},
    'yolo-hrnet': {},
    'mp-pose':
    {'model_complexity': 2, 'min_detection_confidence': 0.5, 'min_tracking_confidence': 0.5},
    'mp-holistic': {
        'ext': 'jpg',
        'force': True,
        'model_complexity': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'mp-handl': {
        'model_complexity': 1,
        'min_detection_confidence': 0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    },
    'mp-handr': {
        'model_complexity': 1,
        'min_detection_confidence': 0.3,
        'min_tracking_confidence': 0.1,
        'static_image_mode': False,
    },
}
        
def lmk_dataset(subject):
    motions = sorted(os.listdir(join(args.path, subject)))
    for motion in motions:
        if os.path.isdir(join(args.path, subject, motion)):
            image_root = join(args.path, subject, motion, 'images')
            annot_root = join(args.path, subject, motion, 'skeletons')
        
            if os.path.exists(annot_root):
                # check the number of annots and images
                if len(os.listdir(image_root)) == len(os.listdir(annot_root)):
                    print('skip ', annot_root)
                # shutil.rmtree(annot_root)
                
            from easymocap.estimator.yolohrnet_wrapper import extract_yolo_hrnet
            extract_yolo_hrnet(image_root, annot_root, "jpg", config['yolo'], config['hrnet'])



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    args = parser.parse_args()
    
    subjects = sorted(os.listdir(args.path))
    print("CPU count: ", mp.cpu_count())
    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        pool.map(partial(lmk_dataset), subjects)
        pool.close()
        pool.join()
        