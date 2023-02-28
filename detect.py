# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import time

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        chunking=False, # chunking image inference
):
    # 帧率检测添加
    # tt = time.time()

    source = str(source)    # 'imgs_detect'
    save_img = not nosave and not source.endswith('.txt')  # save inference images 是否保存检测
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)    # source是否为文件
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))   # source是否为url
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 判断source是否摄像头输入
    screenshot = source.lower().startswith('screen')    # 是否source为屏幕
    if is_url and is_file:  # source为url或文件时进行处理
        source = check_file(source)  # download

    # 当chunking时，默认不保存预测结果label
    save_txt = save_txt and not chunking

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run 预测文件夹路径（是否递增）
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 是否创建预测结果label文件夹

    # Load model
    device = select_device(device)  # 选择推理设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)   # 加载模型
    stride, names, pt = model.stride, model.names, model.pt     # stride: 步长(图像必须给该倍数); names: 标签; pt: 是否为pt权重文件
    imgsz = check_img_size(imgsz, s=stride)  # check image size 检查imgsz参数是否符合stride的标准，不满足则调整

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        multi = 6 if chunking else 1
        # 加载需要推理的图像，stride为模型采样步长，图像需要扩展为stride的倍数才能能输入模型
        dataset = LoadImages(source, img_size=imgsz, stride=multi*stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup 训练热身次数
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # path: 文件绝对路径; im: 处理后的图像(修改为imgsz大小); im0s: 原图像; vid_cap: 视频流的截取; s: 图像处理进展log
    for path, im, im0s, vid_cap, s in dataset:
        # loading image
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 图像读取到设备中,，由ndarray -> tensor
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 降幅电参数和缓冲转换为半浮点(half)数据类型
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim [3, 640, 640] -> [1, 3, 640, 640]

        # Inference
        with dt[1]:
            # visualize参数决定是否存储特征图
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 模型推理
            pred = model(im, augment=augment, visualize=visualize, chunking=chunking)

        # NMS
        if not chunking:
            with dt[2]:
                # 进行NMS算法处理
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 检测后图片保存路径
            save_path = str(save_dir / p.name)  # im.jpg
            # save_path = r'runs/detect/' + p.name
            # 检测后结果保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # txt_path = r'runs/detect/' + p.stem
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 判断是否需要保存检测框
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det) and not chunking:
                # Rescale boxes from img_size to im0 size 预测信息x、y、w、h、c、classes
                # scale_boxes将640*640信息映射到227*227上
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # unique挑出独立不重复种类
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results xyxy是检测框左上角和右下角坐标
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 生成原图的xywh比例
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        # 添加预测框
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        # 单独保存到检测文件夹下面的crops文件夹内
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            elif chunking:
                annotator.chunking_mask(det, alpha=0.3)
            # Stream results
            # 检测后的图像
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 图片上侧添加黑边
            # im0 = cv2.copyMakeBorder(im0, 25, 0, 0, 0, cv2.BORDER_CONSTANT)

            # 添加帧率检测
            # cv2.putText(im0, "FPS:{:.1f} {}ms".format(1. / (time.time() - tt), time.time() - tt), (5, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 235), 2)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold '
                                                                       '置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold '
                                                                      'IOU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image '
                                                                  '单张图片最多检测数目')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results '
                                                                '实时显示检测结果')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt '
                                                                '保存检测结果为txt文件')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels '
                                                                 '保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes '
                                                                 '模型检测的物体裁剪保存')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos '
                                                              '不保存预测结果')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3 '
                                                               '只检测某一一类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS '
                                                                    '跨类别nms')
    parser.add_argument('--augment', action='store_true', help='augmented inference '
                                                               '增强')
    parser.add_argument('--visualize', action='store_true', help='visualize features '
                                                                 '保存特征图')
    parser.add_argument('--update', action='store_true', help='update all models '
                                                              '对模型进行strip_optimizer操作，去除pt中优化器')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name '
                                                                        '预测结果保存路径')
    parser.add_argument('--name', default='exp', help='save results to project/name '
                                                      '预测结果保存文件夹名字')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not incrbnement '
                                                                '如果指定文件夹存在则不新建文件夹')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels) '
                                                                      '预测框线条粗细')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels '
                                                                                  '隐藏标签')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences '
                                                                                '隐藏置信度')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference '
                                                            '使用FP16半精度推理')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference '
                                                           '使用opencv dnn进行onnx推理')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride '
                                                                  '视频流检测步长')
    parser.add_argument('--chunking', action='store_true', help='chunking image to detect')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
