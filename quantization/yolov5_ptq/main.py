# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import time
import torch
import torchvision

import glob
from pathlib import Path
import os, math

import cv2
import onnx
import logging
import argparse
import numpy as np
from PIL import Image
from scipy import special
import yaml

from utils.metrics import box_iou
from utils.augmentations import letterbox
from utils.general import scale_boxes

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--data_path',
    type=str,
    help="Path of COCO dataset, it contains val2017 and annotations subfolder"
)
parser.add_argument(
    '--model_path',
    type=str,
    help="Pre-trained model on onnx file"
)
parser.add_argument(
    '--benchmark',
    action='store_true', \
    default=False
)
parser.add_argument(
    '--tune',
    action='store_true', \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    '--config',
    type=str,
    help="config yaml path"
)
parser.add_argument(
    '--output_model',
    type=str,
    help="output model path"
)
parser.add_argument(
    '--mode',
    type=str,
    help="benchmark mode of performance or accuracy"
)
parser.add_argument(
    '--inference',
    action='store_true', \
    default=True,
    help="whether do inference"
)
args = parser.parse_args()

# def get_anchors(anchors_path, tiny=False):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = np.array(anchors.split(','), dtype=np.float32)
#     return anchors.reshape(3, 3, 2)

# IMAGE_INPUT_SZIE = 416
# ANCHORS = get_anchors("./yolov4_anchors.txt")
# STRIDES = np.array([8, 16, 32])
# XYSCALE = [1.2, 1.1, 1.05]

# class Dataloader:
#     def __init__(self, root, img_dir='val2017', \
#             anno_dir='annotations/instances_val2017.json', filter=None):
#         import json
#         import os
#         import numpy as np
#         from pycocotools.coco import COCO
#         from neural_compressor.experimental.metric.coco_label_map import category_map
#         self.batch_size = 1
#         self.image_list = []
#         img_path = os.path.join(root, img_dir)
#         anno_path = os.path.join(root, anno_dir)
#         coco = COCO(anno_path)
#         img_ids = coco.getImgIds()
#         cat_ids = coco.getCatIds()
#         for idx, img_id in enumerate(img_ids):
#             img_info = {}
#             bboxes = []
#             labels = []
#             ids = []
#             img_detail = coco.loadImgs(img_id)[0]
#             ids.append(img_detail['file_name'].encode('utf-8'))
#             pic_height = img_detail['height']
#             pic_width = img_detail['width']

#             ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
#             anns = coco.loadAnns(ann_ids)
#             for ann in anns:
#                 bbox = ann['bbox']
#                 if len(bbox) == 0:
#                     continue
#                 bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
#                 labels.append(category_map[ann['category_id']].encode('utf8'))
#             img_file = os.path.join(img_path, img_detail['file_name'])
#             if not os.path.exists(img_file) or len(bboxes) == 0:
#                 continue

#             if filter and not filter(None, bboxes):
#                 continue
#             label = [np.array([bboxes]), np.array([labels]), np.zeros((1,0)), np.array([img_detail['file_name'].encode('utf-8')])]
#             with Image.open(img_file) as image:
#                 image = image.convert('RGB')
#                 image, label = self.preprocess((image, label))
#             self.image_list.append((image, label))

#     def __iter__(self):
#         for item in self.image_list:
#             yield item

#     def preprocess(self, sample):
#         image, label = sample
#         image = np.array(image)
#         ih = iw = IMAGE_INPUT_SZIE
#         h, w, _ = image.shape

#         scale = min(iw/w, ih/h)
#         nw, nh = int(scale * w), int(scale * h)
#         image_resized = cv2.resize(image, (nw, nh))

#         image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
#         dw, dh = (iw - nw) // 2, (ih-nh) // 2
#         image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
#         image_padded = image_padded / 255.

#         gt_boxes, str_labels, int_labels, image_ids = label
#         return image_padded[np.newaxis, ...].astype(np.float32), \
#             (gt_boxes, str_labels, int_labels, image_ids, (h, w))

# class Post:
#     def __init__(self) -> None:
#         self.ANCHORS = ANCHORS
#         self.STRIDES = STRIDES
#         self.XYSCALE = XYSCALE
#         self.input_size = IMAGE_INPUT_SZIE

#     def __call__(self, sample):
#         preds, labels = sample
#         labels = labels[0]

#         pred_bbox = postprocess_bbbox(preds, self.ANCHORS, self.STRIDES, self.XYSCALE)
#         bboxes = postprocess_boxes(pred_bbox, labels[4], self.input_size, 0.25)
#         if len(bboxes) == 0:
#             return (np.zeros((1,0,4)), np.zeros((1,0)), np.zeros((1,0))), labels[:4]
#         bboxes_ = np.array(nms(bboxes, 0.63, method='nms'))
#         bboxes, scores, classes = bboxes_[:, :4], bboxes_[:, 4], bboxes_[:, 5]

#         bboxes = np.reshape(bboxes, (1, -1, 4))
#         classes = np.reshape(classes, (1, -1)).astype('int64') + 1
#         scores = np.reshape(scores, (1, -1))
#         return (bboxes, classes, scores), labels[:4]

# def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
#     '''define anchor boxes'''
#     for i, pred in enumerate(pred_bbox):
#         conv_shape = pred.shape
#         output_size = conv_shape[1]
#         conv_raw_dxdy = pred[:, :, :, :, 0:2]
#         conv_raw_dwdh = pred[:, :, :, :, 2:4]
#         xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
#         xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

#         xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
#         xy_grid = xy_grid.astype(np.float)

#         pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
#         pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
#         pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

#     pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
#     pred_bbox = np.concatenate(pred_bbox, axis=0)
#     return pred_bbox

# def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
#     '''remove boundary boxs with a low detection probability'''
#     valid_scale=[0, np.inf]
#     pred_bbox = np.array(pred_bbox)

#     pred_xywh = pred_bbox[:, 0:4]
#     pred_conf = pred_bbox[:, 4]
#     pred_prob = pred_bbox[:, 5:]

#     # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
#     pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
#                                 pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
#     # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
#     org_h, org_w = org_img_shape
#     resize_ratio = min(input_size / org_w, input_size / org_h)

#     dw = (input_size - resize_ratio * org_w) / 2
#     dh = (input_size - resize_ratio * org_h) / 2

#     pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
#     pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

#     # # (3) clip some boxes that are out of range
#     pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
#                                 np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
#     invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
#     pred_coor[invalid_mask] = 0

#     # # (4) discard some invalid boxes
#     bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
#     scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

#     # # (5) discard some boxes with low scores
#     classes = np.argmax(pred_prob, axis=-1)
#     scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
#     score_mask = scores > score_threshold
#     mask = np.logical_and(scale_mask, score_mask)
#     coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
#     return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

# def bboxes_iou(boxes1, boxes2):
#     '''calculate the Intersection Over Union value'''
#     boxes1 = np.array(boxes1)
#     boxes2 = np.array(boxes2)

#     boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
#     boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

#     left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
#     right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

#     inter_section = np.maximum(right_down - left_up, 0.0)
#     inter_area    = inter_section[..., 0] * inter_section[..., 1]
#     union_area    = boxes1_area + boxes2_area - inter_area
#     ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

#     return ious

# def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
#     """
#     :param bboxes: (xmin, ymin, xmax, ymax, score, class)

#     Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
#           https://github.com/bharatsingh430/soft-nms
#     """
#     classes_in_img = list(set(bboxes[:, 5]))
#     best_bboxes = []

#     for cls in classes_in_img:
#         cls_mask = (bboxes[:, 5] == cls)
#         cls_bboxes = bboxes[cls_mask]

#         while len(cls_bboxes) > 0:
#             max_ind = np.argmax(cls_bboxes[:, 4])
#             best_bbox = cls_bboxes[max_ind]
#             best_bboxes.append(best_bbox)
#             cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
#             iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
#             weight = np.ones((len(iou),), dtype=np.float32)

#             assert method in ['nms', 'soft-nms']

#             if method == 'nms':
#                 iou_mask = iou > iou_threshold
#                 weight[iou_mask] = 0.0

#             if method == 'soft-nms':
#                 weight = np.exp(-(1.0 * iou ** 2 / sigma))

#             cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
#             score_mask = cls_bboxes[:, 4] > 0.
#             cls_bboxes = cls_bboxes[score_mask]

#     return best_bboxes


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            logger.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        # LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
        print("--img-size   must be multiple of max stride  ")
    return new_size


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return path, im, im0, self.cap, s

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)





if __name__ == "__main__":
    args.model_path = r"D:\vbox\yolov5m.onnx"
    args.data_path = r"D:\gitee\work_2023\yolov5\data\images"
    model = onnx.load(args.model_path)
    # dataloader = Dataloader(args.data_path)
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = dataloader
        evaluator.postprocess = common.Postprocess(Post)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor import options
        from neural_compressor.experimental import Quantization, common
        options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.eval_dataloader = dataloader
        quantize.calib_dataloader = dataloader
        quantize.postprocess = common.Postprocess(Post)
        q_model = quantize()
        q_model.save(args.output_model)
        
    if args.inference:
        import onnxruntime

        w = args.model_path
        print(w)
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        meta = session.get_modelmeta().custom_metadata_map
        stride = 32
        if "stride" in meta:
            stride, names = int(meta["stride"]), eval(meta["names"])
        device = torch.device("cpu")
        if device.type == "cpu": device = torch.device("cuda:0")

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        dataset = LoadImages(args.data_path, img_size=(640, 640), stride=stride, auto=None, vid_stride=1)
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3: im = im[None]
            im = im.cpu().numpy()
            pred = session.run(output_names, {session.get_inputs()[0].name: im})
            pred = non_max_suppression(pred, 0.25, 0.45, None, max_det=1000)
            seen = 0
            for i, det in enumerate(pred):
                seen += 1
                p, im0, frame = path[i], im0s.copy(), getattr(dataset, "frame", 0)
                # save_path = str("." / p.name)
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]} {'s' * (n > 1)}, "
                        print(s)