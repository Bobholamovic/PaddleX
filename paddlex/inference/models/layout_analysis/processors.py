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

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from ....utils.deps import function_requires_deps, is_dep_available
from ...utils.benchmark import benchmark
from ..object_detection.processors import check_containment, nms

if is_dep_available("opencv-contrib-python"):
    import cv2

Boxes = List[dict]
Number = Union[int, float]


@function_requires_deps("opencv-contrib-python")
def postprocess_mask(mask, epsilon_ratio=0.004):
    """
    Postprocess mask by removing small noise.
    Args:
        mask (ndarray): The input mask of shape [H, W].
        epsilon_ratio (float): The ratio of epsilon.
    Returns:
        ndarray: The output mask after postprocessing.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return mask

    cnt = max(cnts, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
    mask_result = np.zeros_like(mask)
    cv2.drawContours(mask_result, [approx_cnt], -1, 1, thickness=cv2.FILLED)

    return mask_result


@function_requires_deps("opencv-contrib-python")
def extract_masks_from_boxes(boxes, masks, scale_ratio):
    """
    Extract binary mask from boxes.
    Args:
        boxes (ndarray): The bounding boxes of shape [N, 5].
        masks (ndarray): The segmentation masks of shape [N, H, W].
        scale_ratio (tuple): The scale ratio of width and height.
    Returns:
        new_masks (list): The extracted binary masks.
    """
    scale_w, scale_h = scale_ratio
    scale_w /= 4
    scale_h /= 4
    N = len(boxes)
    h, w = masks.shape[1:]
    new_masks = []

    for i in range(N):
        x_min, y_min, x_max, y_max = boxes[i, 2:6]
        img_w = int(x_max - x_min)
        img_h = int(y_max - y_min)

        x_min_s = int(round(x_min * scale_w))
        y_min_s = int(round(y_min * scale_h))
        x_max_s = int(round(x_max * scale_w))
        y_max_s = int(round(y_max * scale_h))

        x_min_s = max(0, min(x_min_s, w - 1))
        x_max_s = max(0, min(x_max_s, w))
        y_min_s = max(0, min(y_min_s, h - 1))
        y_max_s = max(0, min(y_max_s, h))

        if x_max_s <= x_min_s or y_max_s <= y_min_s or img_w <= 0 or img_h <= 0:
            new_masks.append(np.zeros((1, 1), dtype=np.uint8))
            continue

        cropped_mask = masks[i, y_min_s:y_max_s, x_min_s:x_max_s]
        if cropped_mask.size == 0:
            new_masks.append(np.zeros((1, 1), dtype=np.uint8))
            continue

        resized_mask = cv2.resize(
            cropped_mask.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        post_mask = postprocess_mask(resized_mask)
        new_masks.append(post_mask)

    return new_masks


def restructured_boxes(
    boxes: ndarray,
    labels: List[str],
    img_size: Tuple[int, int],
    masks: ndarray = None,
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[2:]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        res = {
            "cls_id": int(box[0]),
            "label": labels[int(box[0])],
            "score": float(box[1]),
            "coordinate": [xmin, ymin, xmax, ymax],
        }
        if masks is not None:
            mask = masks[idx]
            res["mask"] = mask
        box_list.append(res)

    return box_list


def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding boxes from (x1, y1, x2, y2) format using an unclipping ratio.

    Parameters:
    - boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    - unclip_ratio: tuple of (width_ratio, height_ratio), optional.

    Returns:
    - expanded_boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    """
    if unclip_ratio is None:
        return boxes

    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]

                width = x2 - x1
                height = y2 - y1

                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                expanded_boxes.append([class_id, score, new_x1, new_y1, new_x2, new_y2])
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)

    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]

        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        return expanded_boxes


@benchmark.timeit
class LayoutAnalysisProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(
        self, labels: Optional[List[str]] = None, scale_size: Optional[List[int]] = None
    ) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.labels = labels
        self.scale_size = scale_size

    def apply(
        self,
        boxes: ndarray,
        img_size: Tuple[int, int],
        threshold: Union[float, dict],
        layout_nms: Optional[bool],
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        layout_merge_bboxes_mode: Optional[Union[str, dict]],
        masks: Optional[ndarray] = None,
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
            if masks is not None:
                masks = masks[expect_boxes, ...]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            if masks is not None:
                category_filtered_masks = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                if masks is not None:
                    category_masks = masks[boxes[:, 0] == cat_id]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] > category_threshold) & (
                    category_boxes[:, 0] > -1
                )
                if masks is not None:
                    category_masks = category_masks[selected_indices]
                    category_filtered_masks.append(category_masks)
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )
            if masks is not None:
                masks = (
                    np.concatenate(category_filtered_masks)
                    if category_filtered_masks
                    else np.array([])
                )
        if masks is not None:
            scale_ratio = [h / s for h, s in zip(self.scale_size, img_size)]
            masks = extract_masks_from_boxes(boxes, masks, scale_ratio)

        if layout_nms:
            selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes = np.array(boxes[selected_indices])
            if masks is not None:
                masks = [masks[i] for i in selected_indices]

        filter_large_image = True
        # boxes.shape[1] == 6 is object detection, 7 is new ordered object detection, 8 is ordered object detection
        if filter_large_image and len(boxes) > 1 and boxes.shape[1] in [6, 7, 8]:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = self.labels.index("image") if "image" in self.labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            filtered_masks = []
            for idx, box in enumerate(boxes):
                (
                    label_index,
                    score,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                        if masks is not None:
                            filtered_masks.append(masks[idx])
                else:
                    filtered_boxes.append(box)
                    if masks is not None:
                        filtered_masks.append(masks[idx])
            if len(filtered_boxes) == 0:
                filtered_boxes = boxes
                if masks is not None:
                    filtered_masks = masks
            boxes = np.array(filtered_boxes)
            if masks is not None:
                masks = filtered_masks

        if layout_merge_bboxes_mode:
            formula_index = (
                self.labels.index("formula") if "formula" in self.labels else None
            )
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"

                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    contains_other, contained_by_other = check_containment(
                        boxes[:, :6], formula_index
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes = boxes[contained_by_other == 0]
                        if masks is not None:
                            masks = [
                                mask
                                for i, mask in enumerate(masks)
                                if contained_by_other[i] == 0
                            ]
                    elif layout_merge_bboxes_mode == "small":
                        boxes = boxes[(contains_other == 0) | (contained_by_other == 1)]
                        if masks is not None:
                            masks = [
                                mask
                                for i, mask in enumerate(masks)
                                if (contains_other[i] == 0)
                                | (contained_by_other[i] == 1)
                            ]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in [
                        "union",
                        "large",
                        "small",
                    ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            contains_other, contained_by_other = check_containment(
                                boxes[:, :6],
                                formula_index,
                                category_index,
                                mode=layout_mode,
                            )
                            # Remove boxes that are contained by other boxes
                            keep_mask &= contained_by_other == 0
                        elif layout_mode == "small":
                            contains_other, contained_by_other = check_containment(
                                boxes[:, :6],
                                formula_index,
                                category_index,
                                mode=layout_mode,
                            )
                            # Keep boxes that do not contain others or are contained by others
                            keep_mask &= (contains_other == 0) | (
                                contained_by_other == 1
                            )
                boxes = boxes[keep_mask]
                if masks is not None:
                    masks = [mask for i, mask in enumerate(masks) if keep_mask[i]]

        if boxes.size == 0:
            return np.array([])

        if boxes.shape[1] == 8:
            # Sort boxes by their order
            sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks

        if boxes.shape[1] == 7:
            # Sort boxes by their order
            sorted_idx = np.argsort(boxes[:, 6])
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks

        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or  Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
                )
            boxes = unclip_boxes(boxes, layout_unclip_ratio)

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(boxes, self.labels, img_size, masks)
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for data, output in zip(datas, batch_outputs):
            if "masks" in output:
                masks = output["masks"]
            else:
                masks = None
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
                masks,
            )
            outputs.append(boxes)
        return outputs
