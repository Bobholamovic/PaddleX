# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import font_colormap, get_colormap


def draw_box(img: Image.Image, boxes: List[dict]) -> Image.Image:
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.018 * int(img.width)) + 2
    font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")

    draw_thickness = int(max(img.size) * 0.002)
    draw = ImageDraw.Draw(img)
    label2color = {}
    catid2fontcolor = {}
    color_list = get_colormap(rgb=True)

    for i, dt in enumerate(boxes):
        # clsid = dt["cls_id"]
        label, bbox, score = dt["label"], dt["coordinate"], dt["score"]
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
            catid2fontcolor[label] = font_colormap(color_index)
        color = tuple(label2color[label])
        font_color = tuple(catid2fontcolor[label])

        if len(bbox) == 4:
            # draw bbox of normal object detection
            xmin, ymin, xmax, ymax = bbox
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        else:
            raise ValueError(
                f"Only support bbox format of [xmin,ymin,xmax,ymax] or [x1,y1,x2,y2,x3,y3,x4,y4], got bbox of shape {len(bbox)}."
            )

        # draw bbox
        draw.line(
            rectangle,
            width=draw_thickness,
            fill=color,
        )

        # draw label
        text = "{} {:.2f}".format(dt["label"], score)
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top + 4
        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

        text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
        if int(img.width) - bbox[2] < font_size:
            text_position = (
                int(bbox[2] - font_size * 1.1),
                bbox[1] - font_size // 2,
            )
        draw.text(text_position, str(i + 1), font=font, fill="red")

    return img


def restore_to_draw_masks(img_size, boxes):
    """
    Restores extracted masks to the original shape and draws them on a blank image.

    """

    restored_masks = []

    for i, box_info in enumerate(boxes):
        restored_mask = np.zeros(img_size, dtype=np.uint8)
        x_min = int(np.floor(box_info["coordinate"][0]))
        y_min = int(np.floor(box_info["coordinate"][1]))
        h, w = box_info["mask"].shape
        x_max = x_min + w
        y_max = y_min + h
        restored_mask[y_min:y_max, x_min:x_max] = box_info["mask"]
        restored_masks.append(restored_mask)

    return np.array(restored_masks)


def draw_mask(im, boxes, img_size):
    """
    Args:
        im (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
        np_masks (np.ndarray): shape:[N, im_h, im_w]
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_colormap(rgb=True)
    w_ratio = 0.4
    alpha = 0.5
    im = np.array(im).astype("float32")
    clsid2color = {}
    np_masks = restore_to_draw_masks(img_size, boxes)
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]
    for i in range(len(np_masks)):
        clsid, score = int(boxes[i]["cls_id"]), boxes[i]["score"]
        mask = np_masks[i]
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = color_list[color_index]
        color_mask = clsid2color[clsid]
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype("uint8"))


class LayoutAnalysisResult(BaseCVResult):

    def _to_img(self) -> Image.Image:
        """apply"""
        boxes = self["boxes"]
        image = Image.fromarray(self["input_img"][..., ::-1])
        ori_img_size = list(image.size)[::-1]
        if len(boxes) > 0 and "mask" in boxes[0]:
            image = draw_mask(image, boxes, ori_img_size)
        return {"res": draw_box(image, boxes)}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
