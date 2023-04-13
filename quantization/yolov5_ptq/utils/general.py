# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile

import cv2
import IPython
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

# from utils import TryExcept, emojis
# from utils.downloads import gsutil_getsize
# from utils.metrics import box_iou, fitness



def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    # Unzip a *.zip file to path/, excluding files containing strings in exclude list
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


# def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1, retry=3):
#     # Multithreaded file download and unzip function, used in data.yaml for autodownload
#     def download_one(url, dir):
#         # Download 1 file
#         success = True
#         if os.path.isfile(url):
#             f = Path(url)  # filename
#         else:  # does not exist
#             f = dir / Path(url).name
#             LOGGER.info(f'Downloading {url} to {f}...')
#             for i in range(retry + 1):
#                 if curl:
#                     s = 'sS' if threads > 1 else ''  # silent
#                     r = os.system(
#                         f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -')  # curl download with retry, continue
#                     success = r == 0
#                 else:
#                     torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
#                     success = f.is_file()
#                 if success:
#                     break
#                 elif i < retry:
#                     LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
#                 else:
#                     LOGGER.warning(f'❌ Failed to download {url}...')

#         if unzip and success and (f.suffix == '.gz' or is_zipfile(f) or is_tarfile(f)):
#             LOGGER.info(f'Unzipping {f}...')
#             if is_zipfile(f):
#                 unzip_file(f, dir)  # unzip
#             elif is_tarfile(f):
#                 os.system(f'tar xf {f} --directory {f.parent}')  # unzip
#             elif f.suffix == '.gz':
#                 os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
#             if delete:
#                 f.unlink()  # remove zip

#     dir = Path(dir)
#     dir.mkdir(parents=True, exist_ok=True)  # make directory
#     if threads > 1:
#         pool = ThreadPool(threads)
#         pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
#         pool.close()
#         pool.join()
#     else:
#         for u in [url] if isinstance(url, (str, Path)) else url:
#             download_one(u, dir)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


# def non_max_suppression(
#         prediction,
#         conf_thres=0.25,
#         iou_thres=0.45,
#         classes=None,
#         agnostic=False,
#         multi_label=False,
#         labels=(),
#         max_det=300,
#         nm=0,  # number of masks
# ):
#     """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

#     Returns:
#          list of detections, on (n,6) tensor per image [xyxy, conf, cls]
#     """

#     if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
#         prediction = prediction[0]  # select only inference output

#     device = prediction.device
#     mps = 'mps' in device.type  # Apple MPS
#     if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
#         prediction = prediction.cpu()
#     bs = prediction.shape[0]  # batch size
#     nc = prediction.shape[2] - nm - 5  # number of classes
#     xc = prediction[..., 4] > conf_thres  # candidates

#     # Checks
#     assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#     assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

#     # Settings
#     # min_wh = 2  # (pixels) minimum box width and height
#     max_wh = 7680  # (pixels) maximum box width and height
#     max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#     time_limit = 0.5 + 0.05 * bs  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
#     merge = False  # use merge-NMS

#     t = time.time()
#     mi = 5 + nc  # mask start index
#     output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence

#         # Cat apriori labels if autolabelling
#         if labels and len(labels[xi]):
#             lb = labels[xi]
#             v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#             v[:, :4] = lb[:, 1:5]  # box
#             v[:, 4] = 1.0  # conf
#             v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
#             x = torch.cat((x, v), 0)

#         # If none remain process next image
#         if not x.shape[0]:
#             continue

#         # Compute conf
#         x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#         # Box/Mask
#         box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
#         mask = x[:, mi:]  # zero columns if no masks

#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#             x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
#         else:  # best class only
#             conf, j = x[:, 5:mi].max(1, keepdim=True)
#             x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

#         # Filter by class
#         if classes is not None:
#             x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]

#         # Check shape
#         n = x.shape[0]  # number of boxes
#         if not n:  # no boxes
#             continue
#         elif n > max_nms:  # excess boxes
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
#         else:
#             x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy

#         output[xi] = x[i]
#         if mps:
#             output[xi] = output[xi].to(device)
#         if (time.time() - t) > time_limit:
#             LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
#             break  # time limit exceeded

#     return output


# def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
#     # Strip optimizer from 'f' to finalize training, optionally save as 's'
#     x = torch.load(f, map_location=torch.device('cpu'))
#     if x.get('ema'):
#         x['model'] = x['ema']  # replace model with ema
#     for k in 'optimizer', 'best_fitness', 'ema', 'updates':  # keys
#         x[k] = None
#     x['epoch'] = -1
#     x['model'].half()  # to FP16
#     for p in x['model'].parameters():
#         p.requires_grad = False
#     torch.save(x, s or f)
#     mb = os.path.getsize(s or f) / 1E6  # filesize
#     LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


# def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr('evolve: ')):
#     evolve_csv = save_dir / 'evolve.csv'
#     evolve_yaml = save_dir / 'hyp_evolve.yaml'
#     keys = tuple(keys) + tuple(hyp.keys())  # [results + hyps]
#     keys = tuple(x.strip() for x in keys)
#     vals = results + tuple(hyp.values())
#     n = len(keys)

#     # Download (optional)
#     if bucket:
#         url = f'gs://{bucket}/evolve.csv'
#         if gsutil_getsize(url) > (evolve_csv.stat().st_size if evolve_csv.exists() else 0):
#             os.system(f'gsutil cp {url} {save_dir}')  # download evolve.csv if larger than local

#     # Log to evolve.csv
#     s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
#     with open(evolve_csv, 'a') as f:
#         f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

#     # Save yaml
#     with open(evolve_yaml, 'w') as f:
#         data = pd.read_csv(evolve_csv, skipinitialspace=True)
#         data = data.rename(columns=lambda x: x.strip())  # strip keys
#         i = np.argmax(fitness(data.values[:, :4]))  #
#         generations = len(data)
#         f.write('# YOLOv5 Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
#                 f'# Last generation: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) +
#                 '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
#         yaml.safe_dump(data.loc[i][7:].to_dict(), f, sort_keys=False)

#     # Print to screen
#     LOGGER.info(prefix + f'{generations} generations finished, current result:\n' + prefix +
#                 ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
#                                                                                          for x in vals) + '\n\n')

#     if bucket:
#         os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to YOLO outputs
    # Example model = torchvision.models.__dict__['efficientnet_b0'](pretrained=True).to(device).eval()
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_boxes(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for a in d:
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# OpenCV Chinese-friendly functions ------------------------------------------------------------------------------------
imshow_ = cv2.imshow  # copy to avoid recursion errors


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


def imwrite(path, im):
    try:
        cv2.imencode(Path(path).suffix, im)[1].tofile(path)
        return True
    except Exception:
        return False


def imshow(path, im):
    imshow_(path.encode('unicode_escape').decode(), im)


cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow  # redefine

# Variables ------------------------------------------------------------------------------------------------------------
