import torch
import torch.nn as nn
import numpy as np
import pdb
import cv2
from mmcv.runner import get_dist_info 
import mmcv

def _gather_feat(feat, ind, mask=None): 
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):  # kernel size is 3 in the paper
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))  # why flatten the feature maps?
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()  # cat is the num of categories

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode_center(
        tl_heat, br_heat, tl_off_c, br_off_c, tl_regr, br_regr, img_meta,
        scale_factor=None, rescale=False, obj=False, direct=False, 
        linear_factor=8.0, K=100, kernel=3, ae_threshold=0.05, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()
    _, inp_h, inp_w = img_meta['img_shape']

    if not obj:
        tl_heat = torch.sigmoid(tl_heat)
        br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    if direct:
        tl_off_c *= linear_factor
        br_off_c *= linear_factor
    else:
        tl_off_c = torch.exp(tl_off_c)
        br_off_c = torch.exp(br_off_c)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    tl_ys1 = tl_ys.view(batch, K, 1)
    tl_xs1 = tl_xs.view(batch, K, 1)
    br_ys1 = br_ys.view(batch, 1, K)
    br_xs1 = br_xs.view(batch, 1, K)

    tl_ys = tl_ys1.expand(batch, K, K)  # expand for combine all possible boxes
    tl_xs = tl_xs1.expand(batch, K, K)
    br_ys = br_ys1.expand(batch, K, K)
    br_xs = br_xs1.expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_off_c = _tranpose_and_gather_feat(tl_off_c, tl_inds)
        br_off_c = _tranpose_and_gather_feat(br_off_c, br_inds)
        tl_off_c = tl_off_c.view(batch, K, 1, 2)
        br_off_c = br_off_c.view(batch, 1, K, 2)
    
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds) 
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_cxs = tl_xs + tl_off_c[..., 0] + tl_regr[..., 0]
        tl_cys = tl_ys + tl_off_c[..., 1] + tl_regr[..., 1]
        br_cxs = br_xs - br_off_c[..., 0] + br_regr[..., 0]
        br_cys = br_ys - br_off_c[..., 1] + br_regr[..., 1]
    
        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]


    # all possible boxes based on top k corners (ignoring class)
    tl_xs *= (inp_w / width)
    tl_ys *= (inp_h / height)
    br_xs *= (inp_w / width)
    br_ys *= (inp_h / height)

    tl_cxs *= (inp_w / width)
    tl_cys *= (inp_h / height)
    br_cxs *= (inp_w / width)
    br_cys *= (inp_h / height)

    x_off = img_meta['border'][2]
    y_off = img_meta['border'][0]

    tl_xs -= torch.Tensor([x_off]).type_as(tl_xs)
    tl_ys -= torch.Tensor([y_off]).type_as(tl_ys)
    br_xs -= torch.Tensor([x_off]).type_as(br_xs)
    br_ys -= torch.Tensor([y_off]).type_as(br_ys)

    tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
    tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
    br_xs *= br_xs.gt(0.0).type_as(br_xs)
    br_ys *= br_ys.gt(0.0).type_as(br_ys)
    
    tl_cxs -= torch.Tensor([x_off]).type_as(tl_cxs)
    tl_cys -= torch.Tensor([y_off]).type_as(tl_cys)
    br_cxs -= torch.Tensor([x_off]).type_as(br_cxs)
    br_cys -= torch.Tensor([y_off]).type_as(br_cys)

    tl_cxs *= tl_cxs.gt(0.0).type_as(tl_cxs)
    tl_cys *= tl_cys.gt(0.0).type_as(tl_cys)
    br_cxs *= br_cxs.gt(0.0).type_as(br_cxs)
    br_cys *= br_cys.gt(0.0).type_as(br_cys)

    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    group_bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
    centers      = torch.stack((tl_cxs, tl_cys, br_cxs, br_cys), dim=3)
    cre          = torch.zeros_like(centers)
    area_bbox   = torch.abs(br_xs  - tl_xs )*torch.abs(tl_ys  - br_ys ) + 1e-16

    ns = torch.ones_like(area_bbox)*2.1#.6
    l_idxs = area_bbox>3500#22500
    ns[l_idxs]=2.4
    
    cre[...,0] = ((ns+1)*group_bboxes[...,0] + (ns-1)*group_bboxes[...,2])/(2*ns)
    cre[...,1] = ((ns+1)*group_bboxes[...,1] + (ns-1)*group_bboxes[...,3])/(2*ns)
    cre[...,2] = ((ns-1)*group_bboxes[...,0] + (ns+1)*group_bboxes[...,2])/(2*ns)
    cre[...,3] = ((ns-1)*group_bboxes[...,1] + (ns+1)*group_bboxes[...,3])/(2*ns)
    
    area_center = torch.abs(br_cxs - tl_cxs)*torch.abs(tl_cys - br_cys)
    #area_bbox   = torch.abs(br_xs  - tl_xs )*torch.abs(tl_ys  - br_ys ) + 1e-16
    area_cre = torch.abs(cre[...,0] - cre[...,2])*torch.abs(cre[...,1] - cre[...,3])
    dists = area_center/area_cre#area_bbox
    
    tl_cx_inds = ((centers[...,0]<=cre[...,0]) | (centers[...,0]>=cre[...,2]))#.unsqueeze(0)
    tl_cy_inds = ((centers[...,1]<=cre[...,1]) | (centers[...,1]>=cre[...,3]))#.unsqueeze(0)
    br_cx_inds = ((centers[...,2]<=cre[...,0]) | (centers[...,2]>=cre[...,2]))#.unsqueeze(0)
    br_cy_inds = ((centers[...,3]<=cre[...,1]) | (centers[...,3]>=cre[...,3]))#.unsqueeze(0)

    ctr_inds = (tl_cx_inds | tl_cy_inds) & (br_cx_inds | br_cy_inds)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2  # scores for all possible boxes

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)  # tl and br should have the same class

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    # tl should be upper and lefter than br
    width_inds = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    scores[cls_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1
    scores[tl_cx_inds] = -1
    scores[tl_cy_inds] = -1
    scores[br_cx_inds] = -1
    scores[br_cy_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    return bboxes, scores, clses


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    #
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    #
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        #
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        #
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        #
        # avoid the error when num_pos is zero
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def _ae_loss(tag0, tag1, mask):  # mask means only consider the loss of positive corner
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()
    #
    tag_mean = (tag0 + tag1) / 2
    #
    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1  # this is pull loss, smaller means tag0 and tag1 are more similiar
    #
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):  # regression loss
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    #
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian  = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = center
    #
    height, width = heatmap.shape[0:2]
    #process the border
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    #
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

