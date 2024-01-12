'''
  @ Date: 2021-01-08 15:05:05
  @ Author: Yuliang Xiu
  @ LastEditors: Yuliang Xiu
  @ LastEditTime: 2021-01-08 15:05:05
  @ FilePath: /EasyMocap/easymocap/dataset/mv1pmf_ioi.py
'''

from os.path import join
import os

from .mv1pmf import MV1PMF
from .base import crop_image

import cv2

from ..mytools import read_annot


class MV1PMF_IOI(MV1PMF):
    
    def __getitem__(self, index: int):
        images, annots = [], []
        for cam in self.cams:
            imgname = self.imagelist[cam][index]
            assert os.path.exists(imgname), imgname
            if self.has2d:
                annname = self.annotlist[cam][index]
                assert os.path.exists(annname), annname
                assert self.imagelist[cam][index].split('.')[0] == self.annotlist[cam][index].split(
                    '.'
                )[0]
                annot = read_annot(annname, self.kpts_type)
            else:
                annot = []
            if not self.no_img:
                img = cv2.imread(imgname)
                images.append(img)
            else:
                img = None
                images.append(None)
            if self.filter2d is not None:
                annot_valid = []
                for ann in annot:
                    if self.filter2d(**ann):
                        annot_valid.append(ann)
                annot = annot_valid
                annot = self.filter2d.nms(annot)
            if self.ret_crop:
                crop_image(img, annot, True, self.config)
            annots.append(annot)
        if self.undis:
            images = self.undistort(images)
            annots = self.undis_det(annots)

        annots = self.select_person(annots, index, self.pid)

        return images, annots


if __name__ == "__main__":
    root = '/home/yxiu/Code/EasyMocap/data/PuzzleIOI/fitting/00145/apose'
    dataset = MV1PMF_IOI(root)
    images, annots = dataset[0]
    print(images, annots)
