import mmcv
import numpy as np
import torch
import cv2

__all__ = ['MaskTransform_cornernet', 'ImageTransform', 'ImageTransform_cornernet', 'BboxTransform', 'BboxTransform_cornernet', 'MaskTransform', 'Numpy2Tensor']


class ImageTransform(object):
      """
      Preprocess an image.
      1. rescale the image to expected size
      2. normalize the image
      3. flip the image (if needed)
      4. pad the image (if needed)
      5. transpose to (c, h, w)
      """

      def __init__(self,
                   mean=(0, 0, 0),
                   std=(1, 1, 1),
                   pixel_scale=1,
                   to_rgb=True,
                   size_divisor=None):
          self.mean = np.array(mean, dtype=np.float32)
          self.std = np.array(std, dtype=np.float32)
          self.pixel_scale = pixel_scale
          self.to_rgb = to_rgb
          self.size_divisor = size_divisor

      def __call__(self, img, scale, flip=False, keep_ratio=True, crop=False):
          if keep_ratio:
              img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
          else:
              img, w_scale, h_scale = mmcv.imresize(img, scale, return_scale=True)
              scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
          img_shape = img.shape
          img = img * float(self.pixel_scale)
          img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
          if flip:
              img = mmcv.imflip(img)
          if self.size_divisor is not None:
              img = mmcv.impad_to_multiple(img, self.size_divisor)
              pad_shape = img.shape
          else:
              pad_shape = img_shape
          img = img.transpose(2, 0, 1)
          return img, img_shape, pad_shape, scale_factor

class ImageTransform_cornernet(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pixel_scale=1,
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pixel_scale = pixel_scale
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True, crop=False):
        if crop:
            h, w, c = img.shape

            nh = int(h*scale)
            nw = int(w*scale)
            img = mmcv.imresize(img, (nw, nh))
            h, w, c = img.shape

            inp_h = h | 127
            inp_w = w | 127
            center = np.array([h // 2, w // 2])
            if flip:
                img = mmcv.imflip(img)
            img, border, offset = crop_image(img, center, [inp_h, inp_w])
            img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            img = img.transpose(2, 0, 1)
            
            return img, border, offset

        '''if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape'''
        #img = mmcv.imnormalize(img, np.array((0, 0, 0), dtype=np.float32), np.array((1.0/float(self.pixel_scale), 1.0/float(self.pixel_scale), 1.0/float(self.pixel_scale)), dtype=np.float32), False)
        #img = img * float(self.pixel_scale)
        h, w, _ = img.shape
        img = mmcv.imresize(img,(511,511))
        ratio = 511.0/float(h)
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        '''if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape'''
        img = img.transpose(2, 0, 1)
        #return img, (511, 511, 3), ratio#, pad_shape, scale_factor
        return img, (511, 511, 3), None, ratio

def crop_image(image, center, size):
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, 3), dtype=np.float32)
    cropped_image[:, :, 0] += 103.53
    cropped_image[:, :, 1] += 116.28
    cropped_image[:, :, 2] += 123.68

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform_cornernet(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        bboxes = np.array(bboxes)
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        return gt_bboxes
        '''if len(gt_bboxes)>0:
        #try:
            gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        #except IndexError:
        #    raise AssertionError(gt_bboxes)
            gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes'''


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        bboxes = np.array(bboxes)
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        if len(gt_bboxes)>0:
        #try:
            gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1])
        #except IndexError:
        #    raise AssertionError(gt_bboxes)
            gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks

class MaskTransform_cornernet(object):
    def __call__(self, masks, new_scale, flip=False):
        masks = [mmcv.imrescale(mask, new_scale, interpolation='nearest')
                 for mask in masks]
        #print(masks[0].shape)
        if flip:
            masks = [mask[:, ::-1] for mask in masks]

        return masks

class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
