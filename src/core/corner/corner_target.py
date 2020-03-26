import torch
import numpy as np
from random import randint
from .kp_utils import gaussian_radius, draw_gaussian
import math


def corner_target(gt_bboxes, gt_labels, feats, imgscale, num_classes=80, direct=False, obj=False, scale=8.0, dcn=False):
    """
    :param gt_bboxes: list of boxes (xmin, ymin, xmax, ymax)
    :param gt_labels: list of labels
    :param featsize:
    :return:
    """
    b, _, h, w = feats.size()
    im_h, im_w = imgscale

    width_ratio = float(w / im_w)
    height_ratio = float(h / im_h)

    gt_tl_corner_heatmap = np.zeros([b, num_classes, h, w]) * 1.0
    gt_br_corner_heatmap = np.zeros([b, num_classes, h, w]) * 1.0

    gt_tl_obj = np.zeros([b, 1, h, w]) * 1.0
    gt_br_obj = np.zeros([b, 1, h, w]) * 1.0

    gt_tl_off_c = np.zeros([b, 2, h, w]) * 1.0
    gt_br_off_c = np.zeros([b, 2, h, w]) * 1.0

    gt_tl_off_c2 = np.zeros([b, 2, h, w]) * 1.0
    gt_br_off_c2 = np.zeros([b, 2, h, w]) * 1.0


    gt_tl_offsets = np.zeros([b, 2, h, w]) * 1.0
    gt_br_offsets = np.zeros([b, 2, h, w]) * 1.0


    for b_id in range(b):
        #match = []
        for box_id in range(len(gt_labels[b_id])):
            tl_x, tl_y, br_x, br_y = gt_bboxes[b_id][box_id]
            c_x = (tl_x + br_x)/2.0
            c_y = (tl_y + br_y)/2.0

            label = gt_labels[b_id][box_id]  # label is between(1,80)

            ftlx = float(tl_x * width_ratio)
            fbrx = float(br_x * width_ratio)
            ftly = float(tl_y * height_ratio)
            fbry = float(br_y * height_ratio)
            fcx  = float(c_x  * width_ratio)
            fcy  = float(c_y  * height_ratio)


            #tl_x_idx = int(min(ftlx, w - 1))
            #br_x_idx = int(min(fbrx, w - 1))
            #tl_y_idx = int(min(ftly, h - 1))
            #br_y_idx = int(min(fbry, h - 1))
            tl_x_idx = int(ftlx)
            br_x_idx = int(fbrx)
            tl_y_idx = int(ftly)
            br_y_idx = int(fbry)

            width = float(br_x - tl_x)
            height = float(br_y - tl_y)

            width = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            radius = gaussian_radius((height, width), min_overlap=0.3)
            radius = max(0, int(radius))
            # radius = 10

            draw_gaussian(gt_tl_corner_heatmap[b_id, label.long() - 1], [tl_x_idx, tl_y_idx], radius)#, mode='tl')
            draw_gaussian(gt_br_corner_heatmap[b_id, label.long() - 1], [br_x_idx, br_y_idx], radius)#, mode='br')
            draw_gaussian(gt_tl_obj[b_id, 0], [tl_x_idx, tl_y_idx], radius)
            draw_gaussian(gt_br_obj[b_id, 0], [br_x_idx, br_y_idx], radius)

            # gt_tl_corner_heatmap[b_id, label.long()-1, tl_x_idx.long(), tl_y_idx.long()] += 1
            # gt_br_corner_heatmap[b_id, label.long()-1, br_x_idx.long(), br_y_idx.long()] += 1

            tl_x_offset = ftlx - tl_x_idx
            tl_y_offset = ftly - tl_y_idx
            br_x_offset = fbrx - br_x_idx
            br_y_offset = fbry - br_y_idx

            if direct:    
                tl_x_off_c  = (fcx - tl_x_idx)/scale
                tl_y_off_c  = (fcy - tl_y_idx)/scale
                br_x_off_c  = (br_x_idx - fcx)/scale
                br_y_off_c  = (br_y_idx - fcy)/scale
            else:
                tl_x_off_c  = np.log(fcx - ftlx)
                tl_y_off_c  = np.log(fcy - ftly)
                br_x_off_c  = np.log(fbrx - fcx)
                br_y_off_c  = np.log(fbry - fcy)

            gt_tl_offsets[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_offset
            gt_tl_offsets[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_offset
            gt_br_offsets[b_id, 0, br_y_idx, br_x_idx] = br_x_offset
            gt_br_offsets[b_id, 1, br_y_idx, br_x_idx] = br_y_offset

            gt_tl_off_c[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_off_c
            gt_tl_off_c[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_off_c
            gt_br_off_c[b_id, 0, br_y_idx, br_x_idx] = br_x_off_c
            gt_br_off_c[b_id, 1, br_y_idx, br_x_idx] = br_y_off_c

            gt_tl_off_c2[b_id, 0, tl_y_idx, tl_x_idx] = np.log(fcx - ftlx)
            gt_tl_off_c2[b_id, 1, tl_y_idx, tl_x_idx] = np.log(fcy - ftly)
            gt_br_off_c2[b_id, 0, br_y_idx, br_x_idx] = np.log(fbrx - fcx)
            gt_br_off_c2[b_id, 1, br_y_idx, br_x_idx] = np.log(fbry - fcy)
    gt_tl_corner_heatmap = torch.from_numpy(gt_tl_corner_heatmap).type_as(feats)
    gt_br_corner_heatmap = torch.from_numpy(gt_br_corner_heatmap).type_as(feats)
    gt_tl_obj = torch.from_numpy(gt_tl_obj).type_as(feats)
    gt_br_obj = torch.from_numpy(gt_br_obj).type_as(feats)
    gt_tl_off_c   = torch.from_numpy(gt_tl_off_c).type_as(feats)
    gt_br_off_c   = torch.from_numpy(gt_br_off_c).type_as(feats)
    gt_tl_off_c2  = torch.from_numpy(gt_tl_off_c2).type_as(feats)
    gt_br_off_c2  = torch.from_numpy(gt_br_off_c2).type_as(feats)
    gt_tl_offsets = torch.from_numpy(gt_tl_offsets).type_as(feats)
    gt_br_offsets = torch.from_numpy(gt_br_offsets).type_as(feats)

    if obj:
        return gt_tl_obj, gt_br_obj, gt_tl_corner_heatmap, gt_br_corner_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c
    else:
        if not dcn:
            return gt_tl_corner_heatmap, gt_br_corner_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c
        else:
            return gt_tl_corner_heatmap, gt_br_corner_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c, gt_tl_off_c2, gt_br_off_c2

