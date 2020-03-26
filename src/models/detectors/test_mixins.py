from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)
import numpy as np
import cv2
import pycocotools.mask as mask_util
import pdb

class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result


class MaskTestMixin_kpt(object):

    def simple_test_mask(self,
                         score_map,
                         corner_offsets,
                         img_meta,
                         det_bboxes,
                         rescale=False):
        '''
        :param semantic_map: semantic map  hxwx80
        :param img_meta:
        :param det_bboxes:
        :param rescale:
        :return:
        '''
        # TODO: solve hardcode
        semantic_map = (score_map>0.4).astype('int')
        h, w, _ = semantic_map.shape
        instance_map = -np.ones_like(semantic_map)
        border_y, border_x = -img_meta['offset']
        ori_h, ori_w, _ = img_meta['ori_shape']
        _, img_h, img_w = img_meta['img_shape']

        for label, bboxes in enumerate(det_bboxes):
            #keepinds = (bboxes[...,-1]>0.4)
            #bboxes = bboxes[keepinds]
            if (len(bboxes)==0) or (semantic_map[...,label].sum()==0):
                continue
            centers = np.array(bboxes)[...,:4]
            centers[..., 0::2] += border_x
            centers[..., 1::2] += border_y
            pixels = semantic_map[..., label]

            #pdb.set_trace()
            if len(bboxes) == 1:
                instance_map[..., label] = pixels - 1
            else:
                for y in range(h):
                    for x in range(w):
                        if pixels[y, x] == 0:
                            continue
                        tl_x = 4 * (x + corner_offsets[label, y, x]) - 1
                        tl_y = 4 * (y + corner_offsets[label + 80, y, x]) - 1
                        br_x = 4 * (x + corner_offsets[label + 160, y, x]) - 1
                        br_y = 4 * (y + corner_offsets[label + 240, y, x]) - 1
                        #pdb.set_trace()
                        instance_map[y, x, label] = KNN_cluster(centers, np.array([tl_x, tl_y, br_x, br_y]))

        #seg_maps = []
        cls_segms = [[] for _ in range(80)]

        for label in range(80):
            map_with_id = instance_map[..., label]
            if map_with_id.max() == -1:
                continue

            for ins_id in range(map_with_id.max()+1):
                seg_map = (map_with_id == ins_id).astype('float32')
                seg_map *= score_map[...,label]
                seg_map = cv2.resize(seg_map, (img_w, img_h))
                seg_map = (seg_map>0.4).astype('int')
                #seg_map = seg_map[border_y:border_y + ori_h, border_x:border_x + ori_w]
                if seg_map.sum()==0:
                    continue
                seg_map = np.uint8(seg_map)
                
                rle = mask_util.encode(np.array(seg_map[:, :, np.newaxis], order='F'))[0]
                #rle['counts'].decode()
                #cls_segms[label].append(rle)
                cls_segms[label].append(seg_map)
        #pdb.set_trace()
        return cls_segms



def KNN_cluster(centers, x):
    '''
    :param centers: Nxd
    :param x: d
    :return: cluster id
    '''
    return ((x - centers) ** 2).sum(1).argmin()
