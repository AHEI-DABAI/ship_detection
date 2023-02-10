import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
from models.common import DetectMultiBackend

from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# 输入和输出均为array, uint8类型, 三通道(H, W, C) BGR模式

@torch.no_grad()
def detect_img(model,
               image,  # 使用opencv的格式去读
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               visualize=False,  # visualize features
               update=False,  # update all models
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               view_img=False,  # show results
               save_crop=False,  # save cropped prediction boxes
               weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               nosave=False,  # do not save images/videos
               ):
    save_img = not nosave  # 只有在nosave为False并且source文件夹不以.txt结尾的时候对图片进行存储
    names = model.names
    # 图片的预处理
    img = letterbox(image)[0]  # 进行resize和pad
    im = img.transpose((2, 0, 1))[::-1]  # 格式的转换HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # np.ascontiguousarray保证内存连续

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:  # 只有三个维度时增加一个维度
        im = im[None]

    pred = model(im, augment=augment, visualize=visualize)  # 得到预测值
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # NMS
    result = 0
    for i, det in enumerate(pred):
        im0 = image.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # 缩放回原来的大小
            for *xyxy, conf, cls in reversed(det):
                if save_img or save_crop or view_img:  # 在图像上绘制预测框和概率
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # 这里label存储的是预测出的类别和对应的概率，如：ore carrier 0.90
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # 在图像上进行绘制，color是一个类，返回了一个有三个元素的tuple，xyxy对应四个坐标，label代表的是类别和概率
        im0 = annotator.result()  # 此时im0变成ndarray
        result = im0
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    return result  # opencv格式 array:(H, W, C)


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # Load model
    device = select_device(device)  # 选择进行检测的设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size imgsz:[640, 640]

    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup self.jit和self.fp16等都用在这里进行模型的预热(warm up)
    return model, device


if __name__ == '__main__':
    image = cv2.imdecode(np.fromfile('./data/test/zidane.jpg', np.uint8), cv2.IMREAD_COLOR)

    model, device = run()
    image_suc = detect_img(model, image, device=device)

    cv2.imwrite('./data/test_results/result1_cv.png', image_suc)
    print(image_suc.shape, image_suc.dtype)
