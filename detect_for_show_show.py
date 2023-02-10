import os
import sys
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (check_file, check_img_size, check_imshow,
                           increment_path, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from pathlib import Path
import numpy as np
import torch
from utils.general import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.files = [path]
        self.nf = 1  # number of files
        self.auto = auto
        self.new_video(path)  # new video

    def __iter__(self):  # 迭代器的开始位置
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:  # self.count相当于一个计数器，计算已经完成检测的图片或视频，self.nf是图片和视频数目的总和
            raise StopIteration
        path = self.files[self.count]  # 利用self.count作为索引，得到下一张图片的地址

        # Read video
        self.mode = 'video'  # 检测模式是视频video
        ret_val, img0 = self.cap.read()
        while not ret_val:
            self.count += 1
            self.cap.release()
            if self.count == self.nf:  # last video
                raise StopIteration
            else:
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()

        self.frame += 1
        s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


@torch.no_grad()
def run(weights='./yolov5s.pt',  # model.pt path(s)
        source='data/images/people.mp4',  # file/dir/URL/glob, 0 for webcam
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)  # 被检测图片的文件夹
    # Load model
    device = select_device(device)  # 选择进行检测的设备

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size imgsz:[640, 640]

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)  # 重点看数据集的制作！！！！！
    bs = 1  # batch_size
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup self.jit和self.fp16等都用在这里进行模型的预热(warm up)
    for path, im, im0s, vid_cap, s in dataset:
        # path是图片的路径，im是经过resize和pad后的图像，im0s是原始的图像，vid_cap是None，s是字符串用于展示图片信息
        # 对每一张图片进行处理，得到可以进行forward的tensor
        im = torch.from_numpy(im).to(device)  # 把numpy转为tensor并传入GPU
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0  # 进行归一化处理
        if len(im.shape) == 3:  # 只有三个维度
            im = im[None]
            # expand for batch dim 增加一个新的维度(bath_size的维度)，
            # 比如此时im:tensor(1, 3, 384, 640) -> (batch_size, channels, w, h)

        # Inference  前向推理
        pred = model(im, augment=augment, visualize=visualize)  # 进行前馈，得到预测值

        # NMS 非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # getattr用于获取对象的属性

            # normalization gain whwh 返回的是一个tensor[width, height, width, height], .eg:[1920, 1080, 1920, 1080]
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # im 是一个ndarray，lw应该是绘制的线的宽度，pil是是否有特殊字符（如中文）
            if len(det):
                # Rescale boxes from img_size to im0 size 由于存在缩放的问题，所以将训练的图片的大小返回到原始的图片大小
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_crop or view_img:  # Add bbox to image在图像上绘制预测框和概率
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # 这里label存储的是预测出的类别和对应的概率，如：ore carrier 0.90
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # 在图像上进行绘制，color是一个类，返回了一个有三个元素的tuple，xyxy对应四个坐标，label代表的是类别和概率

            # Stream results
            im0 = annotator.result()  # 此时im0变成ndarray,其实也就是图片
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    run()
