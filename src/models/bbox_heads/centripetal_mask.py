
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core.corner.corner_target import corner_target
from mmcv.cnn import normal_init

from mmdet.ops import soft_nms, DeformConv, TopPool, BottomPool, LeftPool, RightPool
from mmdet.core import smooth_l1_loss

from mmdet.core.corner.kp_utils import _decode_center

from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class Centripetal_mask(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels, with_mask=False):
        super(Centripetal_mask, self).__init__()
        self.num_classes = num_classes - 1
        self.in_channels = in_channels

        self.tl_out_channels = self.num_classes + 2 + 2  # 2 is the dim for offset map, as there are 2 coordinates, x,y
        self.br_out_channels = self.num_classes + 2 + 2
        
        self.convs = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

        self.with_mask = with_mask

        self._init_layers()

    def _init_layers(self):
        
        self.tl_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
        self.br_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
        self.mid_tl_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
        self.mid_br_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)

        self.tl_offset = nn.Conv2d(2, 18, 1, bias=False)
        self.br_offset = nn.Conv2d(2, 18, 1, bias=False)
        self.mid_tl_offset = nn.Conv2d(2, 18, 1, bias=False)
        self.mid_br_offset = nn.Conv2d(2, 18, 1, bias=False)

        self.tl_pool = TopLeftPool(self.in_channels)
        self.br_pool = BottomRightPool(self.in_channels)
        self.mid_tl_pool = TopLeftPool(self.in_channels)
        self.mid_br_pool = BottomRightPool(self.in_channels)

        self.tl_heat = make_kp_layer(out_dim=self.num_classes)
        self.br_heat = make_kp_layer(out_dim=self.num_classes)

        self.tl_off_c = make_kp_layer(out_dim=2)
        self.br_off_c = make_kp_layer(out_dim=2)
        
        self.tl_off_c_2 = make_kp_layer(out_dim=2)
        self.br_off_c_2 = make_kp_layer(out_dim=2)

        self.tl_off = make_kp_layer(out_dim=2)
        self.br_off = make_kp_layer(out_dim=2)

        # middle supervision

        self.mid_tl_heat = make_kp_layer(out_dim=self.num_classes)
        self.mid_br_heat = make_kp_layer(out_dim=self.num_classes)

        self.mid_tl_off_c = make_kp_layer(out_dim=2)
        self.mid_br_off_c = make_kp_layer(out_dim=2)
        
        self.mid_tl_off_c_2 = make_kp_layer(out_dim=2)
        self.mid_br_off_c_2 = make_kp_layer(out_dim=2)

        self.mid_tl_off = make_kp_layer(out_dim=2)
        self.mid_br_off = make_kp_layer(out_dim=2)

        if self.with_mask:
            for i in range(4):
                self.convs.append(
                    ConvModule(self.in_channels, self.in_channels, 3, padding=1)
                )
                self.mid_convs.append(
                    ConvModule(self.in_channels, self.in_channels, 3, padding=1)
                )
                
            self.conv_logits = nn.Conv2d(self.in_channels, 81, 1)
            self.mid_conv_logits = nn.Conv2d(self.in_channels, 81, 1)

    def init_weights(self):
        """
        TODO: weight init method
        """
        self.tl_heat[-1].bias.data.fill_(-2.19)
        self.br_heat[-1].bias.data.fill_(-2.19)
        self.mid_tl_heat[-1].bias.data.fill_(-2.19)
        self.mid_br_heat[-1].bias.data.fill_(-2.19)
        normal_init(self.tl_offset, std=0.1)
        normal_init(self.tl_fadp  , std=0.01)
        normal_init(self.br_offset, std=0.1)
        normal_init(self.br_fadp  , std=0.01)
        normal_init(self.mid_tl_offset, std=0.1)
        normal_init(self.mid_tl_fadp  , std=0.01)
        normal_init(self.mid_br_offset, std=0.1)
        normal_init(self.mid_br_fadp  , std=0.01)


    def forward_single(self, feats):
        '''tl_result = self.tl_branch(x)
        br_result = self.br_branch(x)'''
        x = feats[-1]
        mask = None
        mask_mid = None        
        if self.with_mask:
            mask = x
            for conv in self.convs:
                mask = conv(mask)
            mask = self.conv_logits(mask)

        tl_pool = self.tl_pool(x)
        tl_heat = self.tl_heat(tl_pool)
        tl_off_c = self.tl_off_c(tl_pool)
        tl_off = self.tl_off(tl_pool)
        tl_offmap = self.tl_offset(tl_off_c.detach())
        x_tl_fadp = self.tl_fadp(tl_pool, tl_offmap)
        tl_off_c_2= self.tl_off_c_2(x_tl_fadp)


        br_pool = self.br_pool(x)
        br_heat = self.br_heat(br_pool)
        br_off_c = self.br_off_c(br_pool)
        br_off = self.br_off(br_pool)
        br_offmap = self.br_offset(br_off_c.detach())
        x_br_fadp = self.br_fadp(br_pool, br_offmap)
        br_off_c_2= self.br_off_c_2(x_br_fadp)

        tl_result = torch.cat([tl_heat, tl_off_c, tl_off_c_2, tl_off], 1)
        br_result = torch.cat([br_heat, br_off_c, br_off_c_2, br_off], 1)

        x = feats[0]
        
        if self.with_mask:
            mask_mid = x
            for conv in self.mid_convs:
                mask_mid = conv(mask_mid)
            mask_mid = self.mid_conv_logits(mask_mid)
        
        tl_pool_mid = self.mid_tl_pool(x)
        tl_heat_mid = self.mid_tl_heat(tl_pool_mid)
        tl_off_c_mid = self.mid_tl_off_c(tl_pool_mid)
        tl_off_mid = self.mid_tl_off(tl_pool_mid)
        tl_offmap_mid = self.mid_tl_offset(tl_off_c_mid.detach())
        x_tl_fadp_mid = self.mid_tl_fadp(tl_pool_mid, tl_offmap_mid)
        tl_off_c_2_mid= self.mid_tl_off_c_2(x_tl_fadp_mid)

        br_pool_mid = self.mid_br_pool(x)
        br_heat_mid = self.mid_br_heat(br_pool_mid)
        br_off_c_mid = self.mid_br_off_c(br_pool_mid)
        br_off_mid = self.mid_br_off(br_pool_mid)
        br_offmap_mid = self.mid_br_offset(br_off_c_mid.detach())
        x_br_fadp_mid = self.mid_br_fadp(br_pool_mid, br_offmap_mid)
        br_off_c_2_mid= self.mid_br_off_c_2(x_br_fadp_mid)

        tl_result_mid = torch.cat([tl_heat_mid, tl_off_c_mid, tl_off_c_2_mid, tl_off_mid], 1)
        br_result_mid = torch.cat([br_heat_mid, br_off_c_mid, br_off_c_2_mid, br_off_mid], 1)

        if self.with_mask:
            return tl_result, br_result, mask, tl_result_mid, br_result_mid, mask_mid
        else:
            return tl_result, br_result, None, tl_result_mid, br_result_mid, None

    def forward(self, feats):
        """
        :param feats: different layer's feature
        :return: the raw results
        """
        feat = feats  # [-1]# we only use the feature of the last layer
        return self.forward_single(feat)

    def loss(self, tl_result, br_result, mask, mid_tl_result, mid_br_result, mid_mask, gt_bboxes, gt_labels, gt_masks, img_metas, cfg, imgscale):
        gt_tl_heatmap, gt_br_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c,\
        gt_tl_off_c2, gt_br_off_c2 = corner_target(gt_bboxes=gt_bboxes, gt_labels=gt_labels, feats=tl_result, imgscale=imgscale, direct=True, scale=1.0, dcn=True)
        # pred_tl_heatmap = _sigmoid(tl_result[:, :self.num_classes, :, :])
        pred_tl_heatmap = tl_result[:, :self.num_classes, :, :].sigmoid()
        pred_tl_off_c   = tl_result[:, self.num_classes:self.num_classes + 2, :, :]
        pred_tl_off_c2  = tl_result[:, self.num_classes+2:self.num_classes+4, :, :]
        pred_tl_offsets = tl_result[:, -2:, :, :]
        # pred_br_heatmap = _sigmoid(br_result[:, :self.num_classes, :, :])
        pred_br_heatmap = br_result[:, :self.num_classes, :, :].sigmoid()
        pred_br_off_c   = br_result[:, self.num_classes:self.num_classes + 2, :, :]
        pred_br_off_c2  = br_result[:, self.num_classes+2:self.num_classes+4, :, :]
        pred_br_offsets = br_result[:, -2:, :, :]

        # mid_pred_tl_heatmap = _sigmoid(mid_tl_result[:, :self.num_classes, :, :])
        mid_pred_tl_heatmap = mid_tl_result[:, :self.num_classes, :, :].sigmoid()
        mid_pred_tl_off_c   = mid_tl_result[:, self.num_classes:self.num_classes + 2, :, :]
        mid_pred_tl_off_c2  = mid_tl_result[:, self.num_classes+2:self.num_classes+4, :, :]
        mid_pred_tl_offsets = mid_tl_result[:, -2:, :, :]
        # mid_pred_br_heatmap = _sigmoid(mid_br_result[:, :self.num_classes, :, :])
        mid_pred_br_heatmap = mid_br_result[:, :self.num_classes, :, :].sigmoid()
        mid_pred_br_off_c   = mid_br_result[:, self.num_classes:self.num_classes + 2, :, :]
        mid_pred_br_off_c2  = mid_br_result[:, self.num_classes+2:self.num_classes+4, :, :]
        mid_pred_br_offsets = mid_br_result[:, -2:, :, :]

        tl_det_loss = det_loss_(pred_tl_heatmap, gt_tl_heatmap) + det_loss_(mid_pred_tl_heatmap, gt_tl_heatmap)
        br_det_loss = det_loss_(pred_br_heatmap, gt_br_heatmap) + det_loss_(mid_pred_br_heatmap, gt_br_heatmap)
        # tl_det_loss = _neg_loss([pred_tl_heatmap, mid_pred_tl_heatmap], gt_tl_heatmap)
        # br_det_loss = _neg_loss([pred_br_heatmap, mid_pred_br_heatmap], gt_br_heatmap)

        det_loss = (tl_det_loss + br_det_loss) / 2.0

        tl_off_mask = gt_tl_heatmap.eq(1).type_as(gt_tl_heatmap)
        br_off_mask = gt_br_heatmap.eq(1).type_as(gt_br_heatmap)


        tl_off_c_loss = off_loss_(pred_tl_off_c, gt_tl_off_c, mask=tl_off_mask) + off_loss_(mid_pred_tl_off_c, gt_tl_off_c,mask=tl_off_mask)
        br_off_c_loss = off_loss_(pred_br_off_c, gt_br_off_c, mask=br_off_mask) + off_loss_(mid_pred_br_off_c, gt_br_off_c,mask=br_off_mask)
        off_c_loss = tl_off_c_loss.sum() / tl_off_mask.sum() + br_off_c_loss.sum() / br_off_mask.sum()
        off_c_loss /= 2.0
        off_c_loss *= 0.05

        tl_off_c2_loss = off_loss_(pred_tl_off_c2, gt_tl_off_c2, mask=tl_off_mask) + off_loss_(mid_pred_tl_off_c2, gt_tl_off_c2,mask=tl_off_mask)
        br_off_c2_loss = off_loss_(pred_br_off_c2, gt_br_off_c2, mask=br_off_mask) + off_loss_(mid_pred_br_off_c2, gt_br_off_c2,mask=br_off_mask)
        off_c2_loss = tl_off_c2_loss.sum() / tl_off_mask.sum() + br_off_c2_loss.sum() / br_off_mask.sum()
        off_c2_loss /= 2.0

        tl_off_loss = off_loss_(pred_tl_offsets, gt_tl_offsets, mask=tl_off_mask) + off_loss_(mid_pred_tl_offsets, gt_tl_offsets,mask=tl_off_mask)
        br_off_loss = off_loss_(pred_br_offsets, gt_br_offsets, mask=br_off_mask) + off_loss_(mid_pred_br_offsets, gt_br_offsets,mask=br_off_mask)
        off_loss = tl_off_loss.sum() / tl_off_mask.sum() + br_off_loss.sum() / br_off_mask.sum()
        off_loss /= 2.0

        mask_loss = 0
        if self.with_mask:
            for b_id in range(len(gt_labels)):
                for mask_id in range(len(gt_labels[b_id])):
                    mask_label = gt_labels[b_id][mask_id]
                    m_pred     = mask[b_id][mask_label]
                    mid_m_pred = mid_mask[b_id][mask_label]
                    m_gt = torch.from_numpy(gt_masks[b_id][mask_id]).float().cuda()
                    mask_loss += F.binary_cross_entropy_with_logits(m_pred, m_gt)
                    mask_loss += F.binary_cross_entropy_with_logits(mid_m_pred, m_gt)
            mask_loss /= mask.size(0)
            mask_loss /= 2.0

        # return dict(det_loss=det_loss, ae_loss=ae_loss, off_loss=off_loss)
        if self.with_mask:
            return dict(det_loss=det_loss, off_c_loss=off_c_loss, off_c2_loss=off_c2_loss, off_loss=off_loss, mask_loss=mask_loss)
        else:
            return dict(det_loss=det_loss, off_c_loss=off_c_loss, off_c2_loss=off_c2_loss, off_loss=off_loss)

    def get_bboxes(self, tl_result, br_result, mask, mid_tl_result, mid_br_result, mid_mask, img_metas, cfg, rescale=False):
        tl_heat = tl_result[:, :self.num_classes, :, :]
        tl_off_c= tl_result[:, self.num_classes+2:self.num_classes+4, :, :]
        tl_regr = tl_result[:, -2:, :, :]
        br_heat = br_result[:, :self.num_classes, :, :]
        br_off_c= br_result[:, self.num_classes+2:self.num_classes+4, :, :]
        br_regr = br_result[:, -2:, :, :]
        #pdb.set_trace()
        if len(tl_heat) == 2:
            img_metas = img_metas[0]

        if isinstance(img_metas, list):
            img_metas_1 = img_metas[0]
        else:
            img_metas_1 = img_metas

        batch_bboxes, batch_scores, batch_clses = _decode_center(tl_heat=tl_heat, br_heat=br_heat, tl_off_c=tl_off_c, br_off_c=br_off_c, tl_regr=tl_regr, br_regr=br_regr, img_meta=img_metas_1)#[0]
        h, w, _ = img_metas_1['ori_shape']
        #h, w, _ = img_metas[0]['ori_shape']        

        scale = img_metas_1['scale']
        #batch_bboxes /= scale

        if len(batch_bboxes) == 2:
            # print('flip')
            batch_bboxes[1, :, [0, 2]] = w - batch_bboxes[1, :, [2, 0]]


        batch_bboxes = batch_bboxes.view([-1, 4]).unsqueeze(0)
        batch_scores = batch_scores.view([-1, 1]).unsqueeze(0)
        batch_clses = batch_clses.view([-1, 1]).unsqueeze(0)
        # pdb.set_trace()
        # assert  len(img_metas)==len(batch_bboxes)
        result_list = []
        for img_id in range(len(img_metas)):
            # pdb.set_trace()
            bboxes = batch_bboxes[img_id]
            scores = batch_scores[img_id]
            clses = batch_clses[img_id]

            scores_n = scores.cpu().numpy()
            idx = scores_n.argsort(0)[::-1]
            idx = torch.Tensor(idx.astype(float)).long()

            bboxes = bboxes[idx].squeeze()
            scores = scores[idx].view(-1)
            clses = clses[idx].view(-1)

            det_num = len(bboxes)

            # img_h, img_w, _ = img_metas[img_id]['img_shape']
            # ori_h, ori_w, _ = img_metas[img_id]['ori_shape']
            # h_scale = float(ori_h) / float(img_h)
            # w_scale = float(ori_w) / float(img_w)

            # bboxes[:,0::2] *= w_scale
            # bboxes[:,1::2] *= h_scale

            '''clses_idx = (clses + 1).long()
            det_idx   = torch.Tensor(np.arange(det_num)).long()
            scores_81 = -1*torch.ones(det_num, self.num_classes + 1).type_as(scores)
            scores_81[det_idx, clses_idx] = scores

            bboxes_scores = torch.cat([bboxes, scores.unsqueeze(-1)], 1)
            nms_bboxes, _ = nms(bboxes_scores, 0.5)
            #nms_bboxes, nms_labels = multiclass_nms(bboxes, scores_81, 0.5, cfg.nms, cfg.max_per_img)

            result_list.append((nms_bboxes, nms_labels))'''
            detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
            keepinds = (detections[:, -1] > -0.1)  # 0.05
            detections = detections[keepinds]
            labels = clses[keepinds]

            areas = (bboxes[:,2] - bboxes[:,0])*(bboxes[:,3] - bboxes[:,1])
            areas = areas[keepinds]

            #pdb.set_trace()
            if scale == 0.8:
                keepinds2 = (areas >= 96**2)
                detections = detections[keepinds2]
                labels = labels[keepinds2]
                topk = 35
            #elif scale == 2.0:
            #    keepinds2 = (areas <= 32**2)
            #    detections = detections[keepinds2]
            #    labels = labels[keepinds2]
            #    topk = 40
            else:
                topk = 100

            
            # idx = detections[:,-1].topk(len(detections))[1]
            # detections = detections[idx]
            # labels = labels[idx]

            out_bboxes = []
            out_labels = []
            # pdb.set_trace()
            for i in range(80):
                keepinds = (labels == i)
                nms_detections = detections[keepinds]
                a = nms_detections.size(0)
                if nms_detections.size(0) == 0:
                    # print('no NMS')
                    continue
                nms_detections, _ = soft_nms(nms_detections, 0.5, 'gaussian', sigma=0.7)
                b = nms_detections.size(0)
                # print(a,b)

                out_bboxes.append(nms_detections)
                out_labels += [i for _ in range(len(nms_detections))]

            if len(out_bboxes) > 0:
                out_bboxes = torch.cat(out_bboxes)
                # out_labels = 1 + torch.Tensor(out_labels)
                out_labels = torch.Tensor(out_labels)
            else:
                out_bboxes = torch.Tensor(out_bboxes).cuda()
                out_labels = torch.Tensor(out_labels)

            # out_labels = 1+torch.Tensor(out_labels)

            # pdb.set_trace()
            if len(out_bboxes) > 0:
                out_bboxes_np = out_bboxes.cpu().numpy()
                out_labels_np = out_labels.cpu().numpy()
                idx = np.argsort(out_bboxes_np[:, -1])[::-1][:topk]  #100
                out_bboxes_np = out_bboxes_np[idx, :]
                out_labels_np = out_labels_np[idx]
                out_bboxes = torch.Tensor(out_bboxes_np).type_as(out_bboxes)
                out_labels = torch.Tensor(out_labels_np).type_as(out_labels)


            # pdb.set_trace()

            result_list.append((out_bboxes, out_labels))
        return result_list


class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):  # pool1, pool2 should be Class name
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_new(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class TopLeftPool(pool):
    def __init__(self, dim):
        super(TopLeftPool, self).__init__(dim, TopPool, LeftPool)


class BottomRightPool(pool):
    def __init__(self, dim):
        super(BottomRightPool, self).__init__(dim, BottomPool, RightPool)


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


def top_pool(x):  # from right to left
    """
    :param x:feature map x, a Tensor
    :return: feature map with the same size as x
    """
    x_p = torch.zeros_like(x)
    x_p[:, :, :, -1] = x[:, :, :, -1]
    _, _, h, w = x.size()
    for col in range(w - 1, -1, -1):
        x_p[:, :, :, col] = x[:, :, :, col:].max(-1)[0]

    return x_p


def left_pool(x):  # from bottom to top
    x_p = torch.zeros_like(x)
    x_p[:, :, -1, :] = x[:, :, -1, :]
    _, _, h, w = x.size()
    for row in range(h - 1, -1, -1):
        x_p[:, :, row, :] = x[:, :, row:, :].max(-2)[0]

    return x_p


def bottom_pool(x):  # from left to right
    x_p = torch.zeros_like(x)
    x_p[:, :, :, 0] = x[:, :, :, 0]
    _, _, h, w = x.size()
    for col in range(1, w):
        x_p[:, :, :, col] = x[:, :, :, 0:col + 1].max(-1)[0]

    return x_p


def right_pool(x):  # from up to bottom
    x_p = torch.zeros_like(x)
    x_p[:, :, 0, :] = x[:, :, 0, :]
    _, _, h, w = x.size()
    for row in range(1, h):
        x_p[:, :, row, :] = x[:, :, 0:row + 1, :].max(-2)[0]

    return x_p


def det_loss_(preds, gt, Epsilon=1e-12):
    # TODO: add Gaussian to gt_heatmap
    # _, t_num = gt.view([gt.size(0), -1]).size()
    pos_weights = (gt == 1.0).type_as(gt)
    neg_weights = torch.pow(1 - gt, 4).type_as(gt)
    pos_loss = -torch.log(preds + Epsilon) * torch.pow(1 - preds, 2) * pos_weights
    neg_loss = -torch.log(1 - preds + Epsilon) * torch.pow(preds, 2) * neg_weights
    # obj_num = pos_weights.sum(-1).sum(-1).sum(-1)
    obj_num = pos_weights.sum()
    # loss = pos_loss.sum(-1).sum(-1).sum(-1)/obj_num + neg_loss.sum(-1).sum(-1).sum(-1)/(t_num-obj_num)
    if obj_num < 1:
        loss = neg_loss.sum()
    else:
        loss = (pos_loss + neg_loss).sum() / obj_num

    return loss


def _neg_loss(preds, gt, Epsilon=1e-12):
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


def off_loss_(preds, target, mask):
    """
    :param preds: pred_offsets
    :param gt:  gt_offsets
    :param mask: denotes where is those corners
    :return: smooth l1 loss of offsets
    """
    mask = (mask.sum(1) > 0).unsqueeze(1).type_as(preds)
    preds *= mask
    target *= mask

    return smooth_l1_loss(preds, target, reduction='none')


def ae_loss_(tl_preds, br_preds, match):
    """
    :param tl_preds: predicted tensor of top-left embedding
    :param br_preds: predicted tensor of bottom-right embedding
    :param match:
    :return: pull loss and push loss
    """
    b = tl_preds.size(0)

    loss = 0
    pull = 0
    push = 0
    for i in range(b):
        # loss += ae_loss_per_image(tl_preds[i], br_preds[i], match[i])
        loss = ae_loss_per_image(tl_preds[i], br_preds[i], match[i])
        pull += loss[0]
        push += loss[1]
    # return loss
    return pull, push


def ae_loss_per_image(tl_preds, br_preds, match, pull_weight=0.25, push_weight=0.25):
    tl_list = torch.Tensor([]).type_as(tl_preds)
    br_list = torch.Tensor([]).type_as(tl_preds)
    me_list = torch.Tensor([]).type_as(tl_preds)
    for m in match:
        tl_y = m[0][0]
        tl_x = m[0][1]
        br_y = m[1][0]
        br_x = m[1][1]
        tl_e = tl_preds[:, tl_y, tl_x]
        br_e = br_preds[:, br_y, br_x]
        tl_list = torch.cat([tl_list, tl_e])
        br_list = torch.cat([br_list, br_e])
        me_list = torch.cat([me_list, ((tl_e + br_e) / 2.0)])

    assert tl_list.size() == br_list.size()

    N = tl_list.size(0)

    if N > 0:
        pull_loss = (torch.pow(tl_list - me_list, 2) + torch.pow(br_list - me_list, 2)).sum() / N
    else:
        pull_loss = 0

    margin = 1
    push_loss = 0
    for i in range(N):
        mask = torch.ones(N, device=tl_preds.device)
        mask[i] = 0
        push_loss += (mask * F.relu(margin - abs(me_list[i] - me_list))).sum()

    if N > 1:
        push_loss /= (N * (N - 1))
    else:
        pass
    '''if N>0:
        N2 = N*(N-1)
        x0 = me_list.unsqueeze(0)
        x1 = me_list.unsqueeze(1)
        push_loss = (F.relu(1 - torch.abs(x0-x1))-1/(N+1e-4))/(N2+1e-4)
        #push_loss -= 1/(N+1e-4)
        #push_loss /= (N2+1e-4)
        push_loss = push_loss.sum()
    else:
        push_loss = 0'''

    return pull_weight * pull_loss, push_weight * push_loss


def make_kp_layer(out_dim, cnv_dim=256, curr_dim=256):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def _sigmoid(x):
    x = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)
    return x

