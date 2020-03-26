import mmcv
import numpy as np
from numpy import random
import pdb

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w),
                                  int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

class KeepRatioCrop(object):

    def __init__(self,
                 random_scales=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),# 1.4),
                 size=(511,511), border=128):
        self.random_scales = random_scales
        self.crop_size = size
        self.border = border

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            scale = random.choice(self.random_scales)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = _get_border(self.border, h)
            w_border = _get_border(self.border, w)

            for i in range(50):
                ctx = np.random.randint(low=w_border, high=w-w_border)
                cty = np.random.randint(low=h_border, high=h-h_border)

                x0, x1 = max(ctx - new_w // 2, 0), min(ctx + new_w // 2, w)
                y0, y1 = max(cty - new_h // 2, 0), min(cty + new_h // 2, h)
                patch = np.array((int(x0), int(y0), int(x1), int(y1)))

                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                               center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                cropped_img = np.zeros((new_h, new_w, 3), dtype=img.dtype)
                cropped_img[:,:,0] += 103.53
                cropped_img[:,:,1] += 116.28
                cropped_img[:,:,2] += 123.68

                left_w, right_w = ctx - x0, x1 - ctx
                top_h, bottom_h = cty - y0, y1 - cty

                # crop image
                cropped_ctx, cropped_cty = new_w // 2, new_h // 2
                x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
                y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
                cropped_img[y_slice, x_slice, :] = img[y0:y1, x0:x1, :]
               
                # crop detections
                cropped_detections = boxes.copy()
                cropped_detections[:, 0:4:2] -= x0
                cropped_detections[:, 1:4:2] -= y0
                cropped_detections[:, 0:4:2] += cropped_ctx - left_w
                cropped_detections[:, 1:4:2] += cropped_cty - top_h
                #print(boxes.shape,'ori')
                
                cropped_detections, labels, keep_inds = _clip_detections(cropped_img, cropped_detections, labels)
                #print(cropped_detections.shape)
                
                #import pdb
                #pdb.set_trace()
                crop_args = (mask, keep_inds, new_h, new_w, y_slice, x_slice, x0, y0, x1, y1)
                
                return cropped_img, cropped_detections, labels, crop_args

def _clip_detections(image, detections, labels):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    labels = labels[keep_inds]
    return detections, labels, keep_inds


class Noise(object):

    def __init__(self, mean=0, std=1, noise_ratio=0):
        self.mean = mean
        self.std = std
        self.noise_ratio = noise_ratio

    def __call__(self, img, boxes, labels):
        if np.random.uniform(0,1) > self.noise_ratio:
            return img, boxes, labels

        h, w, c = img.shape
        noise_value = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise_value
        return img, boxes, labels

class MaskCrop(object):
    def __call__(self, gt_masks, crop_args):
        '''
        :param gt_masks: a list of gt masks(np.ararry)
        :param crop_args:
        :return:
        '''
        keepinds1, keepinds2, new_h, new_w, y_slice, x_slice, x0, y0, x1, y1 = crop_args
        gt_masks = np.stack(gt_masks, 0)
        #print('mask shape', gt_masks.shape)
        #pdb.set_trace()
        gt_masks = gt_masks[keepinds1]
        gt_masks = gt_masks[keepinds2]
        crop_masks = np.zeros([len(gt_masks), new_h, new_w])

        crop_masks[:, y_slice, x_slice] = gt_masks[:, y0:y1, x0:x1]

        return list(crop_masks)


class ExtraAugmentation_cornernet(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 noise=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        #if expand is not None:
        #    self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(KeepRatioCrop())
            #self.transforms.append(RandomCrop(**random_crop))
        if noise is not None:
            self.transforms.append(Noise(**noise))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            if isinstance(transform, KeepRatioCrop):
                img, boxes, labels, crop_args = transform(img, boxes, labels)
            else:
                img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels, crop_args

        
class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 noise=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            #self.transforms.append(KeepRatioCrop())
            self.transforms.append(RandomCrop(**random_crop))
        if noise is not None:
            self.transforms.append(Noise(**noise))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
