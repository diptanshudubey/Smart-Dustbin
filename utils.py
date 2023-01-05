# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions to display the pose detection results."""

from typing import List

import cv2
import numpy as np
from object_detector import Detection

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1.3
_FONT_THICKNESS = 1

_TEXT_COLOR_RED = (0, 0, 255)    # red
_TEXT_COLOR_GREEN = (0, 255, 0)  # green

def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detections: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  final_catagory = None
  crop_img = image.copy()
  for detection in detections:
    # Draw bounding_box
    crop_img = image.copy()
    
    TEXT_COLOR = (255,0, 0)  #blue
    # Draw label color
    category = detection.categories[0]
    class_name = category.label
    
    TEXT_COLOR = _TEXT_COLOR_RED
        
    
        
    
    start_point = detection.bounding_box.left, detection.bounding_box.top
    end_point = detection.bounding_box.right, detection.bounding_box.bottom
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    class_name = category.label
    final_catagory = class_name
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + detection.bounding_box.left,
                     _MARGIN + _ROW_SIZE + detection.bounding_box.top)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, TEXT_COLOR, _FONT_THICKNESS)
                
 
    #print(detection.bounding_box)
    #print(detection.bounding_box.top) #y1
    #print(detection.bounding_box.left)#x1
    #print(detection.bounding_box.right)#x2
    #print(detection.bounding_box.bottom)#y2
    #crop_img = image
    #print(type(image))
    x1, y1, x2, y2 = detection.bounding_box
    #print(x1, y1, x2, y2)
    crop_img = crop_img[y1:y2, x1:x2]
    #x1, y1, x2, y2 = rects[0]
    #crop_img = image[detection.bounding_box.left:detection.bounding_box.top, detection.bounding_box.right:detection.bounding_box.bottom]
    #cv2.imshow('Croped-Number-Plate', crop_img)

  return image, crop_img, final_catagory
