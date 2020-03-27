import sys
import cv2

import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform, MaskTransform_cornernet,
                         Numpy2Tensor, ImageTransform_cornernet, BboxTransform_cornernet)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation, ExtraAugmentation_cornernet, MaskCrop

import cv2
import random

class CustomDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=False,
                 with_crowd=True,
                 with_label=True,
                 with_triple_grey=False, #default no triple-grey op
                 mixup=False,
                 mixup_sampler=np.random.beta,
                 mixup_args=[0.4,0.4],
                 extra_aug=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 cornernet_mode=False,
                 with_maskhead=False,
                 **kwargs):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        # in test mode or not
        self.test_mode = test_mode
        # if apply triple grey op on training imgs
        self.with_triple_grey=with_triple_grey
        # if apply mixup op on training imgs
        self.mixup=mixup
        self.mixup_sampler=mixup_sampler
        self.mixup_args=mixup_args

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.cornernet_mode = cornernet_mode
        self.with_maskhead = with_maskhead

        if self.cornernet_mode:
            self.img_transform = ImageTransform_cornernet(
                size_divisor=self.size_divisor, **self.img_norm_cfg)
            self.bbox_transform = BboxTransform_cornernet()
            self.mask_transform = MaskTransform_cornernet()
        else:
            self.img_transform = ImageTransform(
                size_divisor=self.size_divisor, **self.img_norm_cfg)
            self.bbox_transform = BboxTransform()
            self.mask_transform = MaskTransform()
        
        #self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            if self.cornernet_mode:
                self.extra_aug = ExtraAugmentation_cornernet(**extra_aug)
                self.mask_crop = MaskCrop()
            else:
                self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    def prepare_train_img_(self,idx):
        img_info = self.img_infos[idx]
        if 'COCO_val2014_' in img_info['filename']:
            s = 13
        elif 'COCO_train2014_' in img_info['filename']:
            s = 15
        else:
            s = 0
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename'][s:]))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None
        else:
            proposals = None
            scores = None

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']
        else:
            gt_bboxes_ignore = None
        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            #official version:
            #return None

            gt_bboxes = [[0,0,0,0]]
            if self.with_label:
                gt_labels = [0]
        if self.extra_aug is not None:
            if self.cornernet_mode:
                img, gt_bboxes, gt_labels, crop_args = self.extra_aug(img, gt_bboxes, gt_labels)
            else:
                img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes, gt_labels)
        # apply transforms
        #first step of transform: convert color
        if self.with_triple_grey:
            if random.random()>=0.5:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.merge([gray,gray,gray])
        else:
            pass

        #after color convert,test img color
 
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        if self.with_mask:
            gt_masks = ann['masks']
            if self.cornernet_mode:
                gt_masks = self.mask_crop(gt_masks, crop_args)

        else:
            gt_masks = None 

        return img_info, img, proposals, scores, gt_bboxes, gt_labels, flip, img_scale, gt_bboxes_ignore, gt_masks

    def prepare_train_img(self, idx):
        img_info, img, proposals, scores, gt_bboxes, gt_labels, flip, img_scale, gt_bboxes_ignore, gt_masks = self.prepare_train_img_(idx)
        if self.mixup:
            idx_ = self._rand_another(idx)
            img_info_, img_, proposals_, scores_, gt_bboxes_, gt_labels_, flip_, img_scale_, gt_bboxes_ignore_, gt_masks_ = self.prepare_train_img_(idx_)
            lambd = max(0, min(1, self.mixup_sampler(*self.mixup_args)))
            height = max(img_info['height'], img_info_['height'])
            width = max(img_info['width'], img_info_['width'])
            mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
            mix_img[:img.shape[0], :img.shape[1], :] = img.astype('float32') * lambd
            mix_img[:img_.shape[0], :img_.shape[1], :] += img_.astype('float32') * (1. - lambd)
            
            gt_bboxes = np.vstack((gt_bboxes,gt_bboxes_))
            if self.with_label:
                gt_labels = np.hstack((gt_labels,gt_labels_))
            if self.with_crowd:
                gt_bboxes_ignore = np.vstack((gt_bboxes_ignore, gt_bboxes_ignore_))
            if self.with_mask:
                gt_masks = np.vstack((gt_masks, gt_masks_))     
            img = mix_img
        
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        
        img = img.copy()
        
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor, flip)

        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            if self.cornernet_mode:
                if self.with_maskhead:
                    gt_masks = self.mask_transform(gt_masks, img_shape[:2], flip)
                else:
                    gt_masks = self.mask_transform(gt_masks, (128, 128), flip)
            else:
                gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, flip)

        if self.mixup:
            ori_shape = (height, width, 3)
        else:
            ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data


    def prepare_test_img(self, idx, gt=True):#keep ratio and padding to desired size
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        
        if gt:
            ann = self.get_ann_info(idx)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            if self.with_mask:
                gt_masks  = ann['masks']

        def prepare_single(img, scale, flip):
            _img, border, offset = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio, crop=True)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=_img.shape, scale=scale,
                border=border, offset=offset,
                flip=flip)
            _img = to_tensor(_img)

            return _img, _img_meta

        imgs = []
        img_metas = []

        for scale in [1.0]:
            _img, _img_meta, = prepare_single(img, scale, False)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))

            if self.flip_ratio > 0:
                _img, _img_meta= prepare_single(
                    img, scale, True)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        if not self.with_mask:
            h, w = _img.shape[0:2]
            gt_masks = [np.zeros([h, w])]
        
        if len(gt_labels)==0:
            gt_labels = np.array([-1])
            h,w=_img.shape[0:2]
            gt_masks = [np.zeros([h,w])]
            gt_bboxes = np.array([[0,0,0,0]])
        if gt:
            data['gt_bboxes'] = gt_bboxes
            data['gt_labels'] = gt_labels
            data['gt_masks']  = gt_masks
            data['idx'] = idx
        return data
