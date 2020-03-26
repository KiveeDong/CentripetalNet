import torch.nn as nn
import torch
import mmcv
import cv2
from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result
from collections import OrderedDict
from mmcv.runner import get_dist_info
import numpy as np
import json
from .test_mixins import MaskTestMixin_kpt
from numpy.random import randint

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

@DETECTORS.register_module
class CentripetalNet(BaseDetector, MaskTestMixin_kpt):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CentripetalNet, self).__init__()
        self.backbone = builder.build_backbone(backbone)       
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        if self.bbox_head.with_mask:
            self.mask_head = True

    def init_weights(self, pretrained=None):
        super(CentripetalNet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_masks):
        """
        :param img:
        :param img_metas:
        :param gt_bboxes: (xmin, ymin, xmax, ymax)
        :param gt_labels:
        :return:
        """
        _,_,h,w = img.size()
        imgscale = (h,w)
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas, self.train_cfg,imgscale)
        losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes+1)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

#    def aug_test_old(self, imgs, img_meta, rescale=False):
#        imgs=torch.cat(imgs)
#        x = self.extract_feat(imgs)
#        outs = self.bbox_head(x)
#        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
#        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
#        bbox_results = [
#            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes+1)
#            for det_bboxes, det_labels in bbox_list
#        ]
#        return bbox_results[0]

    def aug_test(self, imgs_l, img_meta, rescale=False, gt_bboxes=None, gt_labels=None, gt_masks=None, idx=None):
        
        img = imgs_l[0][0]
        img_n = img.squeeze().cpu().numpy()
        img_n = np.transpose(img_n, [1,2,0])
        img_n -= img_n.min()
        img_n /= abs(img_n).max()
        img_n *= 255.0
        ms_results=[]
        bboxes = []
        labels = []
        for i in [0]:
        #for i in [0, 2, 4, 6, 8]:
            imgs = torch.cat(imgs_l[i:i+2])
            x = self.extract_feat(imgs)
            outs = self.bbox_head(x)
            
            bbox_inputs = outs + (img_meta[i:i+2], self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
      
            ms_results.append(bbox_list)
            bboxes.append(bbox_list[0][0])
            labels.append(bbox_list[0][1])

        detections = torch.cat(bboxes)#.cpu().numpy()
        labels = torch.cat(labels)#.cpu().numpy()
        
        bbox_list = [(detections, labels)]        

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes+1)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]


