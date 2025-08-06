
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
from PIL import Image, ImageEnhance
import numpy as np
import os
import sys
import cv2
from copy import deepcopy

_MAX_LEVEL = 10.

_INVALID_BOX = [[-1.0, -1.0, -1.0, -1.0]]

def policy_v0():
    policy = [
        [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
        [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
        [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
    ]
    return policy

def policy_v1():
    policy = [
        [('TranslateX_BBox', 0.6, 4), ('Equalize', 0.8, 10)],
        [('TranslateY_Only_BBoxes', 0.2, 2), ('Cutout', 0.8, 8)],
        [('Sharpness', 0.0, 8), ('ShearX_BBox', 0.4, 0)],
        [('ShearY_BBox', 1.0, 2), ('TranslateY_Only_BBoxes', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Color', 1.0, 6)],
        [('Color', 0.0, 0), ('ShearX_Only_BBoxes', 0.8, 4)],
        [('ShearY_Only_BBoxes', 0.8, 2), ('Flip_Only_BBoxes', 0.0, 10)],
        [('Equalize', 0.6, 10), ('TranslateX_BBox', 0.2, 2)],
        [('Color', 1.0, 10), ('TranslateY_Only_BBoxes', 0.4, 6)],
        [('Rotate_BBox', 0.8, 10), ('Contrast', 0.0, 10)],
        [('Cutout', 0.2, 2), ('Brightness', 0.8, 10)],
        [('Color', 1.0, 6), ('Equalize', 1.0, 2)],
        [('Cutout_Only_BBoxes', 0.4, 6), ('TranslateY_Only_BBoxes', 0.8, 2)],
        [('Color', 0.2, 8), ('Rotate_BBox', 0.8, 10)],
        [('Sharpness', 0.4, 4), ('TranslateY_Only_BBoxes', 0.0, 4)],
        [('Sharpness', 1.0, 4), ('SolarizeAdd', 0.4, 4)],
        [('Rotate_BBox', 1.0, 8), ('Sharpness', 0.2, 8)],
        [('ShearY_BBox', 0.6, 10), ('Equalize_Only_BBoxes', 0.6, 8)],
        [('ShearX_BBox', 0.2, 6), ('TranslateY_Only_BBoxes', 0.2, 10)],
        [('SolarizeAdd', 0.6, 8), ('Brightness', 0.8, 10)],
    ]
    return policy

def policy_vtest():
    policy = [[('TranslateX_BBox', 1.0, 4), ('Equalize', 1.0, 10)], ]
    return policy

def policy_v4():
    policy = [
        [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
        [ ('Sharpness', 0.4, 2)],
        [('TranslateY_BBox', 1.0, 8), ('AutoContrast', 0.8, 2)],
        [('AutoContrast', 0.4, 6), ('ShearX_BBox', 0.8, 8),
         ('Brightness', 0.0, 10)],
        [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10),
         ('AutoContrast', 0.6, 0)],
        [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
        [('Equalize', 0.6, 8), ('Solarize', 0.0, 10)],
        [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
        [('Color', 0.8, 4)],
        [('BBox_Cutout', 1.0, 4), ('Cutout', 0.2, 8)],
        [('Equalize', 0.6, 6)],
        [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2),
         ('Brightness', 0.2, 2)],
        [('Solarize', 0.4, 6), ('SolarizeAdd', 0.2, 10)],
        [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
    ]
    return policy

def policy_v5():
    policy = [
        [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
        [('TranslateY_Only_BBoxes', 1.0, 8), ('AutoContrast', 0.8, 2)],
        [('AutoContrast', 0.4, 6), ('ShearX_Only_BBoxes', 0.8, 8),
         ('Brightness', 0.0, 10)],
        [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10),
         ('AutoContrast', 0.6, 0)],
        [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
        [('Equalize', 0.6, 8), ('Solarize', 0.0, 10)],
        [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
        [('Color', 0.8, 4), ('TranslateY_Only_BBoxes', 1.0, 6)],
        [('Cutout_Only_BBoxes', 1.0, 1), ('Cutout', 0.2, 1)],
        [('Equalize', 0.6, 6)],
        [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2),
         ('Brightness', 0.2, 2)],
        [('TranslateY_Only_BBoxes', 0.4, 8), ('Solarize', 0.4, 6),
         ('SolarizeAdd', 0.2, 10)],
        [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
    ]
    return policy
def policy_v2():
    policy = [
        [('Color', 0.0, 6), ('Cutout', 0.6, 8), ('Sharpness', 0.4, 8)],
        [('Rotate_BBox', 0.4, 8), ('Sharpness', 0.4, 2),
         ('Rotate_BBox', 0.8, 10)],
        [('TranslateY_BBox', 1.0, 8), ('AutoContrast', 0.8, 2)],
        [('AutoContrast', 0.4, 6), ('ShearX_BBox', 0.8, 8),
         ('Brightness', 0.0, 10)],
        [('SolarizeAdd', 0.2, 6), ('Contrast', 0.0, 10),
         ('AutoContrast', 0.6, 0)],
        [('Cutout', 0.2, 0), ('Solarize', 0.8, 8), ('Color', 1.0, 4)],
        [('TranslateY_BBox', 0.0, 4), ('Equalize', 0.6, 8),
         ('Solarize', 0.0, 10)],
        [('TranslateY_BBox', 0.2, 2), ('ShearY_BBox', 0.8, 8),
         ('Rotate_BBox', 0.8, 8)],
        [('Cutout', 0.8, 8), ('Brightness', 0.8, 8), ('Cutout', 0.2, 2)],
        [('Color', 0.8, 4), ('TranslateY_BBox', 1.0, 6),
         ('Rotate_BBox', 0.6, 6)],
        [('Rotate_BBox', 0.6, 10), ('Cutout_Only_BBoxes', 1.0, 4), ('Cutout', 0.2, 8)],
        [('Rotate_BBox', 0.0, 0), ('Equalize', 0.6, 6),
         ('ShearY_BBox', 0.6, 8)],
        [('Brightness', 0.8, 8), ('AutoContrast', 0.4, 2),
         ('Brightness', 0.2, 2)],
        [('TranslateY_BBox', 0.4, 8), ('Solarize', 0.4, 6),
         ('SolarizeAdd', 0.2, 10)],
        [('Contrast', 1.0, 10), ('SolarizeAdd', 0.2, 8), ('Equalize', 0.2, 4)],
    ]
    return policy

def policy_v3():
    policy = [
        [('Posterize', 0.8, 2), ('TranslateX_BBox', 1.0, 8)],
        [('BBox_Cutout', 0.2, 10), ('Sharpness', 1.0, 8)],
        [('Rotate_BBox', 0.6, 8), ('Rotate_BBox', 0.8, 10)],
        [('Equalize', 0.8, 10), ('AutoContrast', 0.2, 10)],
        [('SolarizeAdd', 0.2, 2), ('TranslateY_BBox', 0.2, 8)],
        [('Sharpness', 0.0, 2), ('Color', 0.4, 8)],
        [('Equalize', 1.0, 8), ('TranslateY_BBox', 1.0, 8)],
        [('Posterize', 0.6, 2), ('Rotate_BBox', 0.0, 10)],
        [('AutoContrast', 0.6, 0), ('Rotate_BBox', 1.0, 6)],
        [('Equalize', 0.0, 4), ('Cutout', 0.8, 10)],
        [('Brightness', 1.0, 2), ('TranslateY_BBox', 1.0, 6)],
        [('Contrast', 0.0, 2), ('ShearY_BBox', 0.8, 0)],
        [('AutoContrast', 0.8, 10), ('Contrast', 0.2, 10)],
        [('Rotate_BBox', 1.0, 10), ('Cutout', 1.0, 10)],
        [('SolarizeAdd', 0.8, 6), ('Equalize', 0.8, 8)],
    ]
    return policy

def _equal(val1, val2, eps=1e-8):
    return abs(val1 - val2) <= eps

def blend(image1, image2, factor):
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    difference = image2 - image1
    scaled = factor * difference

    temp = image1 + scaled

    if factor > 0.0 and factor < 1.0:
        return temp.astype(np.uint8)

    return np.clip(temp, a_min=0, a_max=255).astype(np.uint8)

def cutout(image, pad_size, replace=0):
    image_height, image_width = image.shape[0], image.shape[1]

    cutout_center_height = np.random.randint(low=0, high=image_height)
    cutout_center_width = np.random.randint(low=0, high=image_width)

    lower_pad = np.maximum(0, cutout_center_height - pad_size)
    upper_pad = np.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = np.maximum(0, cutout_center_width - pad_size)
    right_pad = np.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = np.pad(np.zeros(
        cutout_shape, dtype=image.dtype),
                  padding_dims,
                  'constant',
                  constant_values=1)
    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    image = np.where(
        np.equal(mask, 0),
        np.ones_like(
            image, dtype=image.dtype) * replace,
        image)
    return image.astype(np.uint8)

def solarize(image, threshold=128):
    return np.where(image < threshold, image, 255 - image)

def solarize_add(image, addition=0, threshold=128):
    added_image = image.astype(np.int64) + addition
    added_image = np.clip(added_image, a_min=0, a_max=255).astype(np.uint8)
    return np.where(image < threshold, added_image, image)

def color(image, factor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    degenerate = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return blend(degenerate, image, factor)

def contrast(img, factor):
    img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor)
    return np.array(img)

def brightness(image, factor):
    degenerate = np.zeros_like(image)
    return blend(degenerate, image, factor)

def posterize(image, bits):
    shift = 8 - bits
    return np.left_shift(np.right_shift(image, shift), shift)

def rotate(image, degrees, replace):
    image = wrap(image)
    image = Image.fromarray(image)
    image = image.rotate(degrees)
    image = np.array(image, dtype=np.uint8)
    return unwrap(image, replace)

def random_shift_bbox(image,
                      bbox,
                      pixel_scaling,
                      replace,
                      new_min_bbox_coords=None):
    image_height, image_width = image.shape[0], image.shape[1]
    image_height = float(image_height)
    image_width = float(image_width)

    def clip_y(val):
        return np.clip(val, a_min=0, a_max=image_height - 1).astype(np.int32)

    def clip_x(val):
        return np.clip(val, a_min=0, a_max=image_width - 1).astype(np.int32)

    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = clip_y(image_height * bbox[2])
    max_x = clip_x(image_width * bbox[3])

    bbox_height, bbox_width = (max_y - min_y + 1, max_x - min_x + 1)
    image_height = int(image_height)
    image_width = int(image_width)

    minval_y = clip_y(min_y - np.int32(pixel_scaling * float(bbox_height) /
                                       2.0))
    maxval_y = clip_y(min_y + np.int32(pixel_scaling * float(bbox_height) /
                                       2.0))
    minval_x = clip_x(min_x - np.int32(pixel_scaling * float(bbox_width) / 2.0))
    maxval_x = clip_x(min_x + np.int32(pixel_scaling * float(bbox_width) / 2.0))

    if new_min_bbox_coords is None:
        unclipped_new_min_y = np.random.randint(
            low=minval_y, high=maxval_y, dtype=np.int32)
        unclipped_new_min_x = np.random.randint(
            low=minval_x, high=maxval_x, dtype=np.int32)
    else:
        unclipped_new_min_y, unclipped_new_min_x = (
            clip_y(new_min_bbox_coords[0]), clip_x(new_min_bbox_coords[1]))
    unclipped_new_max_y = unclipped_new_min_y + bbox_height - 1
    unclipped_new_max_x = unclipped_new_min_x + bbox_width - 1

    new_min_y, new_min_x, new_max_y, new_max_x = (
        clip_y(unclipped_new_min_y), clip_x(unclipped_new_min_x),
        clip_y(unclipped_new_max_y), clip_x(unclipped_new_max_x))
    shifted_min_y = (new_min_y - unclipped_new_min_y) + min_y
    shifted_max_y = max_y - (unclipped_new_max_y - new_max_y)
    shifted_min_x = (new_min_x - unclipped_new_min_x) + min_x
    shifted_max_x = max_x - (unclipped_new_max_x - new_max_x)

    new_bbox = np.stack([
        float(new_min_y) / float(image_height), float(new_min_x) /
        float(image_width), float(new_max_y) / float(image_height),
        float(new_max_x) / float(image_width), bbox[4]
    ])

    bbox_content = image[shifted_min_y:shifted_max_y + 1, shifted_min_x:
                         shifted_max_x + 1, :]

    def mask_and_add_image(min_y_, min_x_, max_y_, max_x_, mask, content_tensor,
                           image_):
        mask = np.pad(mask, [[min_y_, (image_height - 1) - max_y_],
                             [min_x_, (image_width - 1) - max_x_], [0, 0]],
                      'constant',
                      constant_values=1)

        content_tensor = np.pad(content_tensor,
                                [[min_y_, (image_height - 1) - max_y_],
                                 [min_x_, (image_width - 1) - max_x_], [0, 0]],
                                'constant',
                                constant_values=0)
        return image_ * mask + content_tensor

    mask = np.zeros_like(image)[min_y:max_y + 1, min_x:max_x + 1, :]
    grey_tensor = np.zeros_like(mask) + replace[0]
    image = mask_and_add_image(min_y, min_x, max_y, max_x, mask, grey_tensor,
                               image)

    mask = np.zeros_like(bbox_content)
    image = mask_and_add_image(new_min_y, new_min_x, new_max_y, new_max_x, mask,
                               bbox_content, image)

    return image.astype(np.uint8), new_bbox

def _clip_bbox(min_y, min_x, max_y, max_x):
    min_y = np.clip(min_y, a_min=0, a_max=1.0)
    min_x = np.clip(min_x, a_min=0, a_max=1.0)
    max_y = np.clip(max_y, a_min=0, a_max=1.0)
    max_x = np.clip(max_x, a_min=0, a_max=1.0)
    return min_y, min_x, max_y, max_x

def _check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_bbox_boundaries(min_coord, max_coord):
        max_coord = np.maximum(max_coord, 0.0 + delta)
        min_coord = np.minimum(min_coord, 1.0 - delta)
        return min_coord, max_coord

    if _equal(height, 0):
        min_y, max_y = _adjust_bbox_boundaries(min_y, max_y)

    if _equal(width, 0):
        min_x, max_x = _adjust_bbox_boundaries(min_x, max_x)

    return min_y, min_x, max_y, max_x

def _scale_bbox_only_op_probability(prob):
    return prob / 3.0

def _apply_bbox_augmentation(image, bbox, augmentation_func, *args):
    image_height = image.shape[0]
    image_width = image.shape[1]

    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = int(image_height * bbox[2])
    max_x = int(image_width * bbox[3])

    max_y = np.minimum(max_y, image_height - 1)
    max_x = np.minimum(max_x, image_width - 1)

    bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

    augmented_bbox_content = augmentation_func(bbox_content, *args)

    augmented_bbox_content = np.pad(
        augmented_bbox_content, [[min_y, (image_height - 1) - max_y],
                                 [min_x, (image_width - 1) - max_x], [0, 0]],
        'constant',
        constant_values=1)

    mask_tensor = np.zeros_like(bbox_content)

    mask_tensor = np.pad(mask_tensor,
                         [[min_y, (image_height - 1) - max_y],
                          [min_x, (image_width - 1) - max_x], [0, 0]],
                         'constant',
                         constant_values=1)
    image = image * mask_tensor + augmented_bbox_content
    return image.astype(np.uint8)

def _concat_bbox(bbox, bboxes):

    bboxes_sum_check = np.sum(bboxes)
    bbox = np.expand_dims(bbox, 0)
    if _equal(bboxes_sum_check, -4):
        bboxes = bbox
    else:
        bboxes = np.concatenate([bboxes, bbox], 0)
    return bboxes

def _apply_bbox_augmentation_wrapper(image, bbox, new_bboxes, prob,
                                     augmentation_func, func_changes_bbox,
                                     *args):
    should_apply_op = (np.random.rand() + prob >= 1)
    if func_changes_bbox:
        if should_apply_op:
            augmented_image, bbox = augmentation_func(image, bbox, *args)
        else:
            augmented_image, bbox = (image, bbox)
    else:
        if should_apply_op:
            augmented_image = _apply_bbox_augmentation(image, bbox,
                                                       augmentation_func, *args)
        else:
            augmented_image = image
    new_bboxes = _concat_bbox(bbox, new_bboxes)
    return augmented_image.astype(np.uint8), new_bboxes

def _apply_multi_bbox_augmentation(image, bboxes, prob, aug_func,
                                   func_changes_bbox, *args):
    new_bboxes = np.array(_INVALID_BOX)

    bboxes = np.array((_INVALID_BOX)) if bboxes.size == 0 else bboxes

    assert bboxes.shape[1] == 5, "bboxes.shape[1] must be 5!!!!"

    wrapped_aug_func = lambda _image, bbox, _new_bboxes: _apply_bbox_augmentation_wrapper(_image, bbox, _new_bboxes, prob, aug_func, func_changes_bbox, *args)

    num_bboxes = bboxes.shape[0]
    idx = 0

    def cond(_idx, _images_and_bboxes):
        return _idx < num_bboxes

    loop_bboxes = deepcopy(bboxes)

    body = lambda _idx, _images_and_bboxes: [
            _idx + 1, wrapped_aug_func(_images_and_bboxes[0],
                                         loop_bboxes[_idx],
                                         _images_and_bboxes[1])]
    while (cond(idx, (image, new_bboxes))):
        idx, (image, new_bboxes) = body(idx, (image, new_bboxes))

    if func_changes_bbox:
        final_bboxes = new_bboxes
    else:
        final_bboxes = bboxes
    return image, final_bboxes

def _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, aug_func,
                                           func_changes_bbox, *args):
    num_bboxes = len(bboxes)
    new_image = deepcopy(image)
    new_bboxes = deepcopy(bboxes)
    if num_bboxes != 0:
        new_image, new_bboxes = _apply_multi_bbox_augmentation(
            new_image, new_bboxes, prob, aug_func, func_changes_bbox, *args)
    return new_image, new_bboxes

def rotate_only_bboxes(image, bboxes, prob, degrees, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, rotate, func_changes_bbox, degrees, replace)

def shear_x_only_bboxes(image, bboxes, prob, level, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, shear_x, func_changes_bbox, level, replace)

def shear_y_only_bboxes(image, bboxes, prob, level, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, shear_y, func_changes_bbox, level, replace)

def translate_x_only_bboxes(image, bboxes, prob, pixels, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, translate_x, func_changes_bbox, pixels, replace)

def translate_y_only_bboxes(image, bboxes, prob, pixels, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, translate_y, func_changes_bbox, pixels, replace)

def flip_only_bboxes(image, bboxes, prob):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob,
                                                  np.fliplr, func_changes_bbox)

def solarize_only_bboxes(image, bboxes, prob, threshold):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, solarize,
                                                  func_changes_bbox, threshold)

def equalize_only_bboxes(image, bboxes, prob):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(image, bboxes, prob, equalize,
                                                  func_changes_bbox)

def cutout_only_bboxes(image, bboxes, prob, pad_size, replace):
    func_changes_bbox = False
    prob = _scale_bbox_only_op_probability(prob)
    return _apply_multi_bbox_augmentation_wrapper(
        image, bboxes, prob, cutout, func_changes_bbox, pad_size, replace)

def _rotate_bbox(bbox, image_height, image_width, degrees):
    image_height, image_width = (float(image_height), float(image_width))

    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    min_y = -int(image_height * (bbox[0] - 0.5))
    min_x = int(image_width * (bbox[1] - 0.5))
    max_y = -int(image_height * (bbox[2] - 0.5))
    max_x = int(image_width * (bbox[3] - 0.5))
    coordinates = np.stack([[min_y, min_x], [min_y, max_x], [max_y, min_x],
                            [max_y, max_x]]).astype(np.float32)
    rotation_matrix = np.stack([[math.cos(radians), math.sin(radians)],
                                [-math.sin(radians), math.cos(radians)]])
    new_coords = np.matmul(rotation_matrix,
                           np.transpose(coordinates)).astype(np.int32)

    min_y = -(float(np.max(new_coords[0, :])) / image_height - 0.5)
    min_x = float(np.min(new_coords[1, :])) / image_width + 0.5
    max_y = -(float(np.min(new_coords[0, :])) / image_height - 0.5)
    max_x = float(np.max(new_coords[1, :])) / image_width + 0.5

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x, bbox[4]])

def rotate_with_bboxes(image, bboxes, degrees, replace):
    image = rotate(image, degrees, replace)

    image_height, image_width = image.shape[:2]
    wrapped_rotate_bbox = lambda bbox: _rotate_bbox(bbox, image_height, image_width, degrees)
    new_bboxes = np.zeros_like(bboxes)
    for idx in range(len(bboxes)):
        new_bboxes[idx] = wrapped_rotate_bbox(bboxes[idx])
    return image, new_bboxes

def translate_x(image, pixels, replace):
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))
    return unwrap(np.array(image), replace)

def translate_y(image, pixels, replace):
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))
    return unwrap(np.array(image), replace)

def _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal):
    pixels = int(pixels)
    min_y = int(float(image_height) * bbox[0])
    min_x = int(float(image_width) * bbox[1])
    max_y = int(float(image_height) * bbox[2])
    max_x = int(float(image_width) * bbox[3])

    if shift_horizontal:
        min_x = np.maximum(0, min_x - pixels)
        max_x = np.minimum(image_width, max_x - pixels)
    else:
        min_y = np.maximum(0, min_y - pixels)
        max_y = np.minimum(image_height, max_y - pixels)

    min_y = float(min_y) / float(image_height)
    min_x = float(min_x) / float(image_width)
    max_y = float(max_y) / float(image_height)
    max_x = float(max_x) / float(image_width)

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x, bbox[4]])

def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)

    image_height, image_width = image.shape[0], image.shape[1]
    wrapped_shift_bbox = lambda bbox: _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)
    new_bboxes = deepcopy(bboxes)
    num_bboxes = len(bboxes)
    for idx in range(num_bboxes):
        new_bboxes[idx] = wrapped_shift_bbox(bboxes[idx])
    return image.astype(np.uint8), new_bboxes

def shear_x(image, level, replace):
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0))
    return unwrap(np.array(image), replace)

def shear_y(image, level, replace):
    image = Image.fromarray(wrap(image))
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, level, 1, 0))
    return unwrap(np.array(image), replace)

def _shear_bbox(bbox, image_height, image_width, level, shear_horizontal):
    image_height, image_width = (float(image_height), float(image_width))

    min_y = int(image_height * bbox[0])
    min_x = int(image_width * bbox[1])
    max_y = int(image_height * bbox[2])
    max_x = int(image_width * bbox[3])
    coordinates = np.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = coordinates.astype(np.float32)

    if shear_horizontal:
        translation_matrix = np.stack([[1, 0], [-level, 1]])
    else:
        translation_matrix = np.stack([[1, -level], [0, 1]])
    translation_matrix = translation_matrix.astype(np.float32)
    new_coords = np.matmul(translation_matrix,
                           np.transpose(coordinates)).astype(np.int32)

    min_y = float(np.min(new_coords[0, :])) / image_height
    min_x = float(np.min(new_coords[1, :])) / image_width
    max_y = float(np.max(new_coords[0, :])) / image_height
    max_x = float(np.max(new_coords[1, :])) / image_width

    min_y, min_x, max_y, max_x = _clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = _check_bbox_area(min_y, min_x, max_y, max_x)
    return np.stack([min_y, min_x, max_y, max_x, bbox[4]])

def shear_with_bboxes(image, bboxes, level, replace, shear_horizontal):
    if shear_horizontal:
        image = shear_x(image, level, replace)
    else:
        image = shear_y(image, level, replace)

    image_height, image_width = image.shape[:2]
    wrapped_shear_bbox = lambda bbox: _shear_bbox(bbox, image_height, image_width, level, shear_horizontal)
    new_bboxes = deepcopy(bboxes)
    num_bboxes = len(bboxes)
    for idx in range(num_bboxes):
        new_bboxes[idx] = wrapped_shear_bbox(bboxes[idx])
    return image.astype(np.uint8), new_bboxes

def autocontrast(image):

    def scale_channel(image):
        lo = float(np.min(image))
        hi = float(np.max(image))

        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            img = np.clip(im, a_min=0, a_max=255.0)
            return im.astype(np.uint8)

        result = scale_values(image) if hi > lo else image
        return result

    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = np.stack([s1, s2, s3], 2)
    return image

def sharpness(image, factor):
    orig_image = image
    image = image.astype(np.float32)
    kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.
    result = cv2.filter2D(image, -1, kernel).astype(np.uint8)

    return blend(result, orig_image, factor)

def equalize(image):

    def scale_channel(im, c):
        im = im[:, :, c].astype(np.int32)
        histo, _ = np.histogram(im, range=[0, 255], bins=256)

        nonzero = np.where(np.not_equal(histo, 0))
        nonzero_histo = np.reshape(np.take(histo, nonzero), [-1])
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            lut = (np.cumsum(histo) + (step // 2)) // step
            lut = np.concatenate([[0], lut[:-1]], 0)
            return np.clip(lut, a_min=0, a_max=255).astype(np.uint8)

        if step == 0:
            result = im
        else:
            result = np.take(build_lut(histo, step), im)

        return result.astype(np.uint8)

    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = np.stack([s1, s2, s3], 2)
    return image

def wrap(image):
    shape = image.shape
    extended_channel = 255 * np.ones([shape[0], shape[1], 1], image.dtype)
    extended = np.concatenate([image, extended_channel], 2).astype(image.dtype)
    return extended

def unwrap(image, replace):
    image_shape = image.shape
    flattened_image = np.reshape(image, [-1, image_shape[2]])

    alpha_channel = flattened_image[:, 3]

    replace = np.concatenate([replace, np.ones([1], image.dtype)], 0)

    alpha_channel = np.reshape(alpha_channel, (-1, 1))
    alpha_channel = np.tile(alpha_channel, reps=(1, flattened_image.shape[1]))

    flattened_image = np.where(
        np.equal(alpha_channel, 0),
        np.ones_like(
            flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = np.reshape(flattened_image, image_shape)
    image = image[:, :, :3]
    return image.astype(np.uint8)

def _cutout_inside_bbox(image, bbox, pad_fraction):
    image_height, image_width = image.shape[0], image.shape[1]
    bbox = np.squeeze(bbox)

    min_y = np.clip(int(float(image_height) * bbox[0]), 0, image_height)
    min_x = np.clip(int(float(image_width) * bbox[1]), 0, image_width)
    max_y = np.clip(int(float(image_height) * bbox[2]), 0, image_height)
    max_x = np.clip(int(float(image_width) * bbox[3]), 0, image_width)

    mean = np.mean(image[min_y:max_y + 1, min_x:max_x + 1], axis=(0, 1))
    box_height = max_y - min_y + 1
    box_width = max_x - min_x + 1
    pad_size_height = int(pad_fraction * (box_height / 2))
    pad_size_width = int(pad_fraction * (box_width / 2))

    cutout_center_height = np.random.randint(min_y, max_y + 1, dtype=np.int32)
    cutout_center_width = np.random.randint(min_x, max_x + 1, dtype=np.int32)

    lower_pad = np.maximum(0, cutout_center_height - pad_size_height)
    upper_pad = np.maximum(
        0, image_height - cutout_center_height - pad_size_height)
    left_pad = np.maximum(0, cutout_center_width - pad_size_width)
    right_pad = np.maximum(0,
                           image_width - cutout_center_width - pad_size_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    if cutout_shape[0] < 0 or cutout_shape[1] < 0:
        print('cutout_shape:', image_height, ' ', image_width, ' ', lower_pad + upper_pad, ' ', left_pad + right_pad, ' ', cutout_center_width, ' ', pad_size_width )
        print('cutout_shape:', cutout_shape)
        print('min max y x:', min_y, ' ', min_x, ' ', max_y, ' ', max_x)
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]

    mask = np.pad(np.zeros(
        cutout_shape, dtype=image.dtype),
                  padding_dims,
                  'constant',
                  constant_values=1)

    mask = np.expand_dims(mask, 2)
    mask = np.tile(mask, [1, 1, 3])
    return mask, mean

def bbox_cutout(image, bboxes, pad_fraction, replace_with_mean):

    def apply_bbox_cutout(image, bboxes, pad_fraction):
        random_index = np.random.randint(0, bboxes.shape[0], dtype=np.int32)
        chosen_bbox = np.take(bboxes, random_index, axis=0)
        mask, mean = _cutout_inside_bbox(image, chosen_bbox, pad_fraction)

        replace = mean if replace_with_mean else [128] * 3

        image = np.where(
            np.equal(mask, 0),
            np.ones_like(
                image, dtype=image.dtype) * replace,
            image).astype(image.dtype)
        return image

    if len(bboxes) != 0:
        image = apply_bbox_cutout(image, bboxes, pad_fraction)

    return image, bboxes

NAME_TO_FUNC = {
        'AutoContrast': autocontrast,
        'Equalize': equalize,
        'Posterize': posterize,
        'Solarize': solarize,
        'SolarizeAdd': solarize_add,
        'Color': color,
        'Contrast': contrast,
        'Brightness': brightness,
        'Sharpness': sharpness,
        'Cutout': cutout,
        'BBox_Cutout': bbox_cutout,
        'Rotate_BBox': rotate_with_bboxes,
        'TranslateX_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
                image, bboxes, pixels, replace, shift_horizontal=True),
        'TranslateY_BBox': lambda image, bboxes, pixels, replace: translate_bbox(
                image, bboxes, pixels, replace, shift_horizontal=False),
        'ShearX_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
                image, bboxes, level, replace, shear_horizontal=True),
        'ShearY_BBox': lambda image, bboxes, level, replace: shear_with_bboxes(
                image, bboxes, level, replace, shear_horizontal=False),
        'Rotate_Only_BBoxes': rotate_only_bboxes,
        'ShearX_Only_BBoxes': shear_x_only_bboxes,
        'ShearY_Only_BBoxes': shear_y_only_bboxes,
        'TranslateX_Only_BBoxes': translate_x_only_bboxes,
        'TranslateY_Only_BBoxes': translate_y_only_bboxes,
        'Flip_Only_BBoxes': flip_only_bboxes,
        'Solarize_Only_BBoxes': solarize_only_bboxes,
        'Equalize_Only_BBoxes': equalize_only_bboxes,
        'Cutout_Only_BBoxes': cutout_only_bboxes,
}

def _randomly_negate_tensor(tensor):
    should_flip = np.floor(np.random.rand() + 0.5) >= 1
    final_tensor = tensor if should_flip else -tensor
    return final_tensor

def _rotate_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate_tensor(level)
    return (level, )

def _shrink_level_to_arg(level):
    if level == 0:
        return (1.0, )
    level = 2. / (_MAX_LEVEL / level) + 0.9
    return (level, )

def _enhance_level_to_arg(level):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1, )

def _shear_level_to_arg(level):
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate_tensor(level)
    return (level, )

def _translate_level_to_arg(level, translate_const):
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate_tensor(level)
    return (level, )

def _bbox_cutout_level_to_arg(level, hparams):
    cutout_pad_fraction = (level /
                           _MAX_LEVEL) * 0.75
    return (cutout_pad_fraction, False)

def level_to_arg(hparams):
    return {
        'AutoContrast': lambda level: (),
        'Equalize': lambda level: (),
        'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4), ),
        'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256), ),
        'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110), ),
        'Color': _enhance_level_to_arg,
        'Contrast': _enhance_level_to_arg,
        'Brightness': _enhance_level_to_arg,
        'Sharpness': _enhance_level_to_arg,
        'Cutout':
        lambda level: (int((level / _MAX_LEVEL) * 100), ),
        'BBox_Cutout': lambda level: _bbox_cutout_level_to_arg(level, hparams),
        'TranslateX_BBox':
        lambda level: _translate_level_to_arg(level, 250),
        'TranslateY_BBox':
        lambda level: _translate_level_to_arg(level, 250),
        'ShearX_BBox': _shear_level_to_arg,
        'ShearY_BBox': _shear_level_to_arg,
        'Rotate_BBox': _rotate_level_to_arg,
        'Rotate_Only_BBoxes': _rotate_level_to_arg,
        'ShearX_Only_BBoxes': _shear_level_to_arg,
        'ShearY_Only_BBoxes': _shear_level_to_arg,
        'TranslateX_Only_BBoxes':
        lambda level: _translate_level_to_arg(level, 120),
        'TranslateY_Only_BBoxes':
        lambda level: _translate_level_to_arg(level, 120),
        'Flip_Only_BBoxes': lambda level: (),
        'Solarize_Only_BBoxes':
        lambda level: (int((level / _MAX_LEVEL) * 256), ),
        'Equalize_Only_BBoxes': lambda level: (),
        'Cutout_Only_BBoxes':
        lambda level: (int((level / _MAX_LEVEL) * 50), ),
    }

def bbox_wrapper(func):

    def wrapper(images, bboxes, *args, **kwargs):
        return (func(images, *args, **kwargs), bboxes)

    return wrapper

def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams):
    func = NAME_TO_FUNC[name]
    args = level_to_arg(augmentation_hparams)[name](level)

    if 'prob' in inspect.getfullargspec(func)[0]:
        args = tuple([prob] + list(args))

    if 'replace' in inspect.getfullargspec(func)[0]:
        assert 'replace' == inspect.getfullargspec(func)[0][-1]
        args = tuple(list(args) + [replace_value])

    if 'bboxes' not in inspect.getfullargspec(func)[0]:
        func = bbox_wrapper(func)
    return (func, prob, args)

def _apply_func_with_prob(func, image, args, prob, bboxes):
    assert isinstance(args, tuple)
    assert 'bboxes' == inspect.getfullargspec(func)[0][1]

    if 'prob' in inspect.getfullargspec(func)[0]:
        prob = 1.0

    should_apply_op = np.floor(np.random.rand() + 0.5) >= 1
    if should_apply_op:
        augmented_image, augmented_bboxes = func(image, bboxes, *args)
    else:
        augmented_image, augmented_bboxes = (image, bboxes)
    return augmented_image, augmented_bboxes

def select_and_apply_random_policy(policies, image, bboxes):
    policy_to_select = np.random.randint(0, len(policies), dtype=np.int32)
    for (i, policy) in enumerate(policies):
        if i == policy_to_select:
            image, bboxes = policy(image, bboxes)
    return (image, bboxes)

def build_and_apply_nas_policy(policies, image, bboxes, augmentation_hparams):
    replace_value = [128, 128, 128]

    tf_policies = []
    for policy in policies:
        tf_policy = []
        for policy_info in policy:
            policy_info = list(
                policy_info) + [replace_value, augmentation_hparams]

            tf_policy.append(_parse_policy_info(*policy_info))
        def make_final_policy(tf_policy_):
            def final_policy(image_, bboxes_):
                for func, prob, args in tf_policy_:
                    image_, bboxes_ = _apply_func_with_prob(func, image_, args,
                                                            prob, bboxes_)
                return image_, bboxes_

            return final_policy

        tf_policies.append(make_final_policy(tf_policy))

    augmented_images, augmented_bboxes = select_and_apply_random_policy(
        tf_policies, image, bboxes)
    return (augmented_images, augmented_bboxes)

def distort_image_with_autoaugment(image, bboxes, augmentation_name):
    available_policies = {
        'v0': policy_v0,
        'v1': policy_v1,
        'v2': policy_v2,
        'v3': policy_v3,
        'v4': policy_v4,
        'v5': policy_v5,
        'test': policy_vtest
    }
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(
            augmentation_name))

    policy = available_policies[augmentation_name]()
    augmentation_hparams = {}
    return build_and_apply_nas_policy(policy, image, bboxes,
                                      augmentation_hparams)
