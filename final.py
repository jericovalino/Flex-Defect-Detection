"""
Naming Style Guide:
  variable,my_variable : variable
  function,my_function : function
  widget,myWidget      : tkinter widgets
  CONSTANT,MY_CONSTANT : constant
"""

from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import threading
from PIL import Image, ImageTk
import tkinter as tk
from datetime import datetime
import cv2
import os
import numpy
import numpy as np
import sys
import tensorflow as tf
# This is needed, for us to import utils since the utils is inside the object_detection folder.
sys.path.append("../models/research")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

FINAL_RESULT = None

CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_frozen_graph = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb") # may change in runtime.
PATH_TO_FROZEN_GRAPH = os.path.join(CWD_PATH, "IG", "frozen_inference_graph.pb") # default(CONSTANT).
# List of the strings that is used to add correct label for each box.
path_to_labels = os.path.join(CWD_PATH, "dataset", "label.pbtxt") # may change in runtime.
PATH_TO_LABELS = os.path.join(CWD_PATH, "dataset", "label.pbtxt") # default(CONSTANT).
# Path to where the capture images will be save.
SAVE_IMAGES_PATH = os.path.join(CWD_PATH, 'save_images')
# Number of classes.
NUM_CLASSES = 1
# Grabs the current timestamp.
ts = datetime.now()

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import abc
import collections
import functools
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualize_boxes_and_masks_and_keypoints(
    image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    original_image_spatial_shape: [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: [N, 3] tensor containing the spatial size of unpadded
      original_image.
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  if instance_masks is not None and keypoints is None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks
    ]
  elif instance_masks is None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores, keypoints
    ]
  elif instance_masks is not None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks, keypoints
    ]
  else:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores
    ]

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example() or
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
      corresponds to detections, while the subimage on the right corresponds to
      groundtruth.
  """
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  if len(eval_dict[detection_fields.detection_classes].shape) == 1:
    for key in eval_dict:
      if key != input_data_fields.original_image:
        eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    instance_masks = None
    if detection_fields.detection_masks in eval_dict:
      instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[detection_fields.detection_masks][indx], axis=0),
          tf.uint8)
    keypoints = None
    if detection_fields.detection_keypoints in eval_dict:
      keypoints = tf.expand_dims(
          eval_dict[detection_fields.detection_keypoints][indx], axis=0)
    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
      groundtruth_instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[input_data_fields.groundtruth_instance_masks][indx],
              axis=0), tf.uint8)

    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_classes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_scores][indx], axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=instance_masks,
        keypoints=keypoints,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
        tf.expand_dims(
            tf.ones_like(
                eval_dict[input_data_fields.groundtruth_classes][indx],
                dtype=tf.float32),
            axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=groundtruth_instance_masks,
        keypoints=None,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_detections_list.append(
        tf.concat([images_with_detections, images_with_groundtruth], axis=2))
  return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  global FINAL_RESULT
  FINAL_RESULT = None
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        FINAL_RESULT = "FAIL"
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)
  return image


def add_cdf_image_summary(values, name):
  """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """
  def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return image
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
  """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """

  def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return image
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(object):
  """Abstract base class responsible for visualizations during evaluation.

  Currently, summary images are not run during evaluation. One way to produce
  evaluation images in Tensorboard is to provide tf.summary.image strings as
  `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
  responsible for accruing images (with overlaid detections and groundtruth)
  and returning a dictionary that can be passed to `eval_metric_ops`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image'):
    """Creates an EvalMetricOpsVisualization.

    Args:
      category_index: A category index (dictionary) produced from a labelmap.
      max_examples_to_draw: The maximum number of example summaries to produce.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and kepoints are in
        normalized coordinates (as opposed to absolute coordiantes).
        Default is True.
      summary_name_prefix: A string prefix for each image summary.
    """

    self._category_index = category_index
    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns metric ops for use in tf.estimator.EstimatorSpec.

    Args:
      eval_dict: A dictionary that holds an image, groundtruth, and detections
        for a batched example. Note that, we use only the first example for
        visualization. See eval_util.result_dict_for_batched_example() for a
        convenient method for constructing such a dictionary. The dictionary
        contains
        fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
        fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
          tensor containing the size of the original image.
        fields.InputDataFields.true_image_shape: [batch_size, 3]
          tensor containing the spatial size of the upadded original image.
        fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
          float32 tensor with groundtruth boxes in range [0.0, 1.0].
        fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
          int64 tensor with 1-indexed groundtruth classes.
        fields.InputDataFields.groundtruth_instance_masks - (optional)
          [batch_size, num_boxes, H, W] int64 tensor with instance masks.
        fields.DetectionResultFields.detection_boxes - [batch_size,
          max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
          1.0].
        fields.DetectionResultFields.detection_classes - [batch_size,
          max_num_boxes] int64 tensor with 1-indexed detection classes.
        fields.DetectionResultFields.detection_scores - [batch_size,
          max_num_boxes] float32 tensor with detection scores.
        fields.DetectionResultFields.detection_masks - (optional) [batch_size,
          max_num_boxes, H, W] float32 tensor of binarized masks.
        fields.DetectionResultFields.detection_keypoints - (optional)
          [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
          keypoints.

    Returns:
      A dictionary of image summary names to tuple of (value_op, update_op). The
      `update_op` is the same for all items in the dictionary, and is
      responsible for saving a single side-by-side image with detections and
      groundtruth. Each `value_op` holds the tf.summary.image string for a given
      image.
    """
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    update_op = tf.py_func(self.add_images, [[images[0]]], [])
    image_tensors = tf.py_func(
        get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right'):
    super(VisualizeSingleFrameDetections, self).__init__(
        category_index=category_index,
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(
        eval_dict, self._category_index, self._max_boxes_to_draw,
        self._min_score_thresh, self._use_normalized_coordinates)


class Application:
    def __init__(self):
        """Initialize application which uses OpenCV + Tkinter"""
        self.cam = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera.
        self.current_frame = None  # current image from the camera.
        self.root = Tk()  # initialize root window.
        self.root.overrideredirect(True) # removes os default window border.
        self.root.attributes("-topmost", True) # makes the window always ontop.

        self.SAVING_DIR = os.path.join(CWD_PATH, "save_images")
        bg_color = "#222"

        # Gets both half the screen width/height and window width/height.
        screen_length_x = int(self.root.winfo_screenwidth()/2 - (848/2))
        screen_length_y = int(self.root.winfo_screenheight()/2 - (665/2))

        self.root.geometry(f"848x665+{screen_length_x}+{screen_length_y}") # window position(center).

        self.root.iconbitmap(default = 'icon.ico') # set icon.
        self.root.title("Flex Defect Detection")  # set window title.
        self.root.configure(bg = bg_color) # bg color
        self.root.protocol('DELETE_WINDOW', self.close_window)


        """declairing variable for GUI starts here"""
        self.topBar = Frame(self.root)
        self.blueLine = Frame(self.root, bg = "#4285F4", height = 7)
        self.titleBar = Frame(self.root, bg = bg_color, height = 30)
        self.centerFrame = Frame(self.root, bg = bg_color)
        self.terminalFrame = Frame(self.root, bg = bg_color, height = 100)
        self.terminalScrollBar = Scrollbar(self.terminalFrame)
        self.terminalListBox = Listbox(self.terminalFrame, bg="#1c313a", fg="#fff", width=103, height=8, borderwidth=0,
            highlightthickness = 0, font = ('verdana', 9), yscrollcommand=self.terminalScrollBar.set)
        self.terminalScrollBar.config(command = self.terminalListBox.yview)
        self.panel = Label(self.centerFrame, width = 91, bg = "#eee", text = "No signal...")  # initialize video panel.
        self.menuFrame = LabelFrame(self.centerFrame, text = "MENU", bg = "#eee")
        logo_img = PhotoImage(file = 'logo.PNG')
        logo = Label(self.titleBar, image = logo_img, bd = 0, bg = bg_color); logo.image = logo_img
        """declairing variable for UI ends here"""


        """declaring variables for top bar buttons starts here"""
        exit_img = PhotoImage(file = "red.PNG")
        min_img = PhotoImage(file = "yellow.PNG")
        max_img = PhotoImage(file = "green.PNG")
        exitButton = Button(self.topBar, image = exit_img, bd=0, command = self.close_window); exitButton.image = exit_img
        maxButton = Button(self.topBar, image = max_img, bd=0, command = self.clear_terminal); maxButton.image = max_img
        minButton = Button(self.topBar, image = min_img, bd=0, command = self.minimize); minButton.image = min_img
        """declaring variables for top bar buttons ends here"""


        """initializing dragging feature starts here"""
        self.drag = self.topBar
        self.drag.bind('<ButtonPress-1>', self.start_move)
        self.drag.bind('<ButtonRelease-1>', self.stop_move)
        self.drag.bind('<B1-Motion>', self.on_motion)
        """initializing dragging feature ends here"""

        # Declaring a Label variable that holds the title text.
        titleLabel = Label(self.titleBar, text="Cosmetic Quality Defect Detection in Electronics",
            bd=0, bg=bg_color, fg='#fefefe',font=('verdana', 10))

        self.topBar.grid(sticky = (W,E))
        exitButton.grid(column = 0, row = 0, padx = (3,0))
        minButton.grid(column = 1, row = 0, padx = (1,0))
        maxButton.grid(column = 2, row = 0, padx = (1,0))     
        self.blueLine.grid(row = 1, column = 0, sticky = (W,E))
        self.titleBar.grid(row = 2, column = 0, sticky = (W,E))
        logo.grid(row = 0, column = 0, pady = 3)
        titleLabel.grid(row = 0, column = 1, padx = 10)
        self.centerFrame.grid(row = 3, column = 0, sticky = (W,E))
        self.panel.grid(row = 0, column = 0, sticky = (N,S,W), padx = (3,0))
        self.menuFrame.grid(row = 0, column = 1, sticky = (N,E,S,W), padx = (0,3))
        self.terminalFrame.grid(row = 4, column = 0, sticky = (W,E))
        self.terminalListBox.grid(row = 0, column = 0, sticky = (N,E,S,W), padx = (3,0), pady = 3)
        self.terminalScrollBar.grid(row = 0, column = 1, sticky = (N,S,E), padx = (1,3), pady = 3)


        # creates a button, that when pressed, will take the current frame and then run the detection.
        self.detectionButton = Button(self.menuFrame, text = "RUN DETECTION", bg = "#e0e0e0",
            width = 23, height = 2, relief = GROOVE, font = ('verdana', 10), command = self.take_snapshot)
        self.detectionButton.grid(column = 0, row = 1, sticky = (E,W))

        # creates a button, that when pressed, will open a fileDialog for user to select saving dir.
        changeDirButton = Button(self.menuFrame, text = "change saving directory",
            relief=GROOVE, command = (self.change_saving_dir))
        changeDirButton.grid(column = 0, row = 3, sticky = (E,W), pady=(295,0))

        # creates a button, that when pressed, will call the funtion self.change_model.
        selectModelButton = Button(self.menuFrame, text = "change trained model",
            relief=GROOVE, command = (self.change_model))
        selectModelButton.grid(column = 0, row = 4, sticky = (E,W))

        # creates a button, that when pressed, will call the funtion self.change_label.
        selectLabelButton = Button(self.menuFrame, text = "change label file",
            relief=GROOVE, command = (self.change_label))
        selectLabelButton.grid(column = 0, row = 5, sticky = (E,W))

        # creates a button, that when pressed, will call the funtion self.load_defaults.
        restoreDefaultButton = Button(self.menuFrame, text = "restore default settings",
            relief = GROOVE, command = (self.load_defaults))
        restoreDefaultButton.grid(column = 0, row = 6, sticky = (E,W))
        
        # create a progress bar for detection process.
        self.progressBar = ttk.Progressbar(self.menuFrame, mode = 'determinate',
            orient = HORIZONTAL, maximum = 4, value = 0)
        self.progressBar.grid(column = 0, row = 0, sticky = (E,W))


        self.counter = 0 #counter for captured image numbering.
        self.listCounter = 1 #counter for terminal listbox (index).

        self.video_loop() # start a self.video_loop .

        self.root.bind('<Map>', self.check_map) # added bindings to pass windows status to function.
        self.root.bind('<Unmap>', self.check_map)

        self.load_defaults() # load default setting at start-up.

    def check_map(self, event):
        """apply override on deiconify"""
        if str(event) == '<Map event>':
            self.root.overrideredirect(True)

    def start_move(self, event):
        """commputes the changes in x and y when top bar is being drag"""
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        """resets x and y value when the mouse stops moving"""
        self.x = None
        self.y = None

    def on_motion(self, event):
        """change the position of the window/s based on the calculation when top bar is being drag"""
        x = (event.x_root - self.x - self.drag.winfo_rootx() + self.drag.winfo_rootx())
        y = (event.y_root - self.y - self.drag.winfo_rooty() + self.drag.winfo_rooty())
        self.root.geometry(f"+{x}+{y}")
        try:
            self.win.attributes('-topmost',True)
            self.win.geometry(f"+{x-5}+{y+23}")
        except:
            pass

    def video_loop(self):
        """get frame from the video stream and show it in panel every 30 milliseconds"""
        ok, self.frame = self.cam.read()  # read frame from video stream.
        if ok:  # frame captured without any errors.
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA.
            self.current_frame = Image.fromarray(cv2image)  # convert image for PIL.
            frametk = ImageTk.PhotoImage(image=self.current_frame)  # convert image for tkinter.
            self.panel.frametk = frametk  # anchor frametk so it does not be deleted by garbage-collector.
            self.panel.config(image = frametk, width = 640, height = 480)  # show the image in panel.
        if not ok:
            frametk = None
            self.panel.frametk = None
            self.clear_terminal()
            self.terminal_print("WARNING", "No connected camera found!!!")
            self.cam = cv2.VideoCapture(0)
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds.

    def terminal_print(self, mtype, message):
        """print something in the terminal(terminalListBox)"""
        time = datetime.now()
        self.terminalListBox.insert(self.listCounter, f"[{time.hour}:{time.minute}:{time.second}] [{mtype}] {message}")
        self.terminalListBox.see(END)
        self.listCounter += 1

    def clear_terminal(self):
        """clears the terminal(terminalListBox)"""
        self.terminalListBox.delete(0, END)
        self.progressBar.config(value = 0) # resets the progress bar

    def close_window(self):
        """Destroy the window and release all resources """
        self.root.destroy() # destroys window.
        self.cam.release() # release web camera.

    def minimize(self):
        """minimize the window"""
        self.root.wm_withdraw()
        self.root.wm_overrideredirect(False)
        self.root.wm_iconify()

    def load_inference_graph(self,path_to_frozen_graph):
        """load the (.pb) file into memory"""
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.terminal_print("INFO", "Loading the Inference Graph into memory. Done")

    def load_label(self,path_to_labels):
        """load the (.pbtxt) file into memory"""
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.terminal_print("INFO", "Loading labels into memory. Done.")

    def take_snapshot(self):
        """Take snapshot, then saves the captured to saving dir"""
        self.counter += 1
        self.detectionButton.config(state = DISABLED)
        FINAL_RESULT = None
        try:
            self.close_result()
        except:
            pass

        captured_image_name = "{}_[{}]_Captured.jpg".format(ts.strftime("%d-%m-%y"),self.counter)  # construct filename
        captured_image_path = os.path.join(self.SAVING_DIR, captured_image_name)  # construct output path
        cv2.imwrite(captured_image_path, self.frame)
        self.terminal_print("SAVED", captured_image_name)

        imageopen = Image.open(captured_image_path)
        raw_image = cv2.imread(captured_image_path)
        image_np = self.load_image_into_numpy_array(imageopen)
        t1 = threading.Thread(target=self.run_inference, args=(raw_image,image_np)).start() # run inference in seperate thread to avoid mainloop freeze
        self.progressBar.config(value=1)

    def load_image_into_numpy_array(self,image):
        """loads images into numpy array that will be feed to the model"""
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def run_inference(self,raw_image,image):
        """runs the detection"""
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                self.terminal_print("INFO", "Evaluating captured image. Please wait...")                
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # add a progress on the progress bar
                self.progressBar.config(value = 2)
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict = {image_tensor: np.expand_dims(image, 0)})
                # add a progress on the progress bar
                self.progressBar.config(value = 3)

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                # Visualization of the results of detection.
                result_image = visualize_boxes_and_labels_on_image_array(
                    raw_image,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    self.category_index,
                    instance_masks = output_dict.get('detection_masks'),
                    use_normalized_coordinates = True,
                    line_thickness = 2)
                self.terminal_print("INFO", "Detection complete.")
                result_image_name = "{}_[{}]_Result.jpg".format(ts.strftime("%d-%m-%y"), self.counter)
                result_image_path = os.path.join(self.SAVING_DIR, result_image_name)
                cv2.imwrite(result_image_path, result_image)
                self.terminal_print("SAVED", result_image_name)
                self.show_result(result_image_path)

    def show_result(self,path):
        """creates a window that displays the image result"""
        # add a progress on the progress bar
        self.progressBar.config(value = 4)
        image_open = Image.open(path)
        result = ImageTk.PhotoImage(image_open)
        self.win = Toplevel()
        if FINAL_RESULT == "FAIL":
            self.win.title("FAIL")
        else:
            self.win.title("PASS")
        self.win.attributes('-topmost', True)
        self.win.focus_force()
        self.win.protocol("WM_DELETE_WINDOW", self.close_result)
        self.win.geometry(f"+{self.root.winfo_x()-5}+{self.root.winfo_y()+23}")
        panel = Label(self.win, image = result); panel.image = result
        panel.pack()
        self.detectionButton.config(state = NORMAL)
    
    def close_result(self):
        self.progressBar.config(value = 0)
        self.win.destroy()

    def change_saving_dir(self):
        """change the path where the captured and relust images will be save"""
        self.SAVING_DIR = filedialog.askdirectory()
        self.terminal_print("PATH", f" {self.SAVING_DIR}")

    def change_model(self):
        """change the path to trained model(.pb file)"""
        path_to_frozen_graph = filedialog.askopenfilename(
            title = "Select Trained Model File(.pb)", filetypes=[("Model File", "*.pb")])
        if not path_to_frozen_graph:
            path_to_frozen_graph = PATH_TO_FROZEN_GRAPH 
        self.terminal_print("PATH", f" {path_to_frozen_graph}")
        self.load_inference_graph(path_to_frozen_graph)

    def change_label(self):
        """change the path to labels(.pbtxt file)"""
        path_to_labels = filedialog.askopenfilename(
            title = "Select Label File(.pbtxt)", filetypes=[("Label File", "*.pbtxt")])
        if not path_to_labels:
            path_to_labels = PATH_TO_LABELS
        self.terminal_print("PATH", f" {path_to_labels}")
        self.load_label(path_to_labels)

    def load_defaults(self):
        """calls the default start up settings and load it to memory"""
        self.terminal_print("INFO", "Load default settings")
        t1 = threading.Thread(target = self.load_inference_graph, args = (PATH_TO_FROZEN_GRAPH,)).start() # run process in seperate thread to avoid mainloop freeze
        t2 = threading.Thread(target = self.load_label, args = (PATH_TO_LABELS,)).start() # run process in seperate thread to avoid mainloop freeze


if __name__ == "__main__":
    # start the app
    run = Application()
    run.root.mainloop()
