# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
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
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 视频流数据集
    if webcam:
        # 检查是否应该显示图像，具体实现可能涉及图像显示库的调用和配置
        view_img = check_imshow(warn=True)
        # 使用 LoadStreams 类加载视频流数据集。该类的具体实现可能涉及视频流的读取和处理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 将批量大小设置为数据集中的样本数
        bs = len(dataset)
    # 屏幕截图数据集
    elif screenshot:
        # 使用 LoadScreenshots 类加载屏幕截图数据集。该类的具体实现可能涉及屏幕截图的读取和处理
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # 除了以上情况的其他情况
    else:
        # 使用 LoadImages 类加载图像数据集。该类的具体实现可能涉及图像的读取和处理。
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # 初始化 vid_path 和 vid_writer 列表，长度为批量大小，并将每个元素初始化为 None。
    # 用于存储视频路径和视频写入器对象。
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference

    # 这是一个模型预热的方法调用。它将输入图像的大小 imgsz 作为参数传递给模型的 warmup() 方法。具体的预热操作可能涉及模型参数初始化、缓存填充等操作，以提高模型在后续推理过程中的性能。
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen 记录已处理的图像数量，windows 用于存储图像窗口信息，dt 是一个元组，包含了三个 Profile 对象，用于记录推理时间。
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # 遍历数据集中的每个样本
    # path 是图像路径，im 是图像数据，im0s 是原始图像数据，vid_cap 是视频捕获对象，s 是图像缩放比例。
    for path, im, im0s, vid_cap, s in dataset:

        # 使用 dt[0] 对象记录下面代码块的执行时间。
        with dt[0]:
            # 将图像数据转换为 PyTorch 张量，并将其移动到模型所在的设备。
            im = torch.from_numpy(im).to(model.device)
            # 如果模型使用 FP16（半精度浮点数）进行推理，则将图像张量转换为半精度浮点数类型；否则，转换为单精度浮点数类型。??
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 将图像张量的值从 0-255 范围归一化到 0.0-1.0 范围。
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 如果图像张量的维度为 3（缺少批次维度），则在第一维度上添加一个批次维度。
            # 如(height, width, channels)缺少批次维度，为了在推理过程中处理这样的张量，需要将其扩展为(1, height, width, channels)的形状，其中1是批次维度。
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # 如果模型使用 XML（多尺度推理）并且图像张量的批次维度大于 1，则将图像张量拆分成多个子张量。这可能涉及将多个尺度的图像作为独立的批次进行推理。
            if model.xml and im.shape[0] > 1:
                # torch.chunk(im, im.shape[0], 0) 的含义是将 im 沿着批次维度（第0维）进行分块，每个块的大小为 batch_size。
                # 返回的结果是一个包含多个子张量的列表，每个子张量的形状为 (1, height, width, channels)
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        # 使用 dt[1] 对象记录下面代码块的执行时间。
        with dt[1]:
            # 根据 visualize 变量的值，确定是否需要保存可视化结果。
            # increment_path() 函数用于生成一个唯一的文件路径，这个路径将用于保存可视化结果。
            # 如果 visualize 为 True，则将生成的路径赋值给 visualize 变量；否则，将 False 赋值给 visualize 变量。
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            """根据模型是否使用了多尺度推理（model.xml）以及图像张量的批次维度是否大于1（im.shape[0] > 1），分两种情况对预测结果进行处理"""
            # 如果模型使用了多尺度推理并且图像张量的批次维度大于1
            if model.xml and im.shape[0] > 1:
                # 有多个尺度的图像需要处理，那么需要对每个子图像进行推理，并将预测结果存储在 pred 变量中。
                pred = None
                # 遍历每个子图像
                for image in ims:
                    # 如果 pred 为 None，说明是第一个子图像，将模型的预测结果添加到 pred 中；
                    if pred is None:
                        # augment 和 visualize 是可选的参数，用于指定是否进行数据增强和可视化。
                        # unsqueeze() 函数用于在指定的维度上增加一个维度。将单个图像的预测结果从 (num_predictions, num_attributes) 的形状扩展为 (1, num_predictions, num_attributes) 的形状。
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        # 首先，model(image, augment=augment, visualize=visualize) 是调用模型对新的图像 image 进行推理的过程，得到新的预测结果。
                        # torch.cat() 函数用于在指定的维度上拼接张量。在这里，使用 dim=0 表示在第0维（批次维度）上进行拼接。
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                # 保持数据结构的一致性，无论是多个尺度的图像还是单个图像，pred 都是一个列表，第一个元素是预测结果张量，第二个元素是 None。????
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        # 使用非最大抑制（non-maximum suppression, NMS）对预测结果进行处理，以过滤掉重叠的边界框并选择置信度最高的边界框。
        # 使用 dt[2] 对象记录下面代码块的执行时间。
        with dt[2]:
            # non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)函数 用于对预测结果 pred 进行非最大抑制。
            # pred，形状为 (num_scales, num_predictions, num_attributes)，其中 num_scales 表示尺度的数量，num_predictions 表示每个尺度的预测数量，num_attributes 表示每个预测的属性数量。
            # conf_thres：置信度阈值，用于过滤掉置信度低于阈值的预测结果。
            # iou_thres：IoU（交并比）阈值，用于合并重叠的边界框。
            # classes：类别列表，用于指定感兴趣的类别。
            # agnostic_nms：布尔值，指示是否使用类别不可知的非最大抑制。
            # max_det：最大检测数量，用于限制输出的边界框数量。
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        # 用于将预测结果写入 CSV 文件的函数。
        # image_name、prediction 和 confidence，用于记录图像名称、预测结果和置信度。
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            # 打开 CSV 文件，使用 mode="a" 表示以追加模式打开文件。
            with open(csv_path, mode="a", newline="") as f:
                # 创建一个 DictWriter 对象，用于将字典数据写入 CSV 文件。
                # fieldnames=data.keys() 指定了 CSV 文件的列名。
                # data.keys() 返回的结果将是一个可迭代对象 dict_keys(["Image Name", "Prediction", "Confidence"])
                writer = csv.DictWriter(f, fieldnames=data.keys())
                # 如果 CSV 文件不存在，则写入列名（即 CSV 文件的表头）。
                if not csv_path.is_file():
                    writer.writeheader()
                # 将 data 中的数据写入 CSV 文件的一行。
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
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
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL") # before change
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train/exp17/weights/best.pt", help="model path or triton URL") # 使用训练后的文件
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--source", type=str, default="C:/Users/bearb/Desktop/GraduationProject/datasets/FiftyTotal_1_include_book/images/val", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    # parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu") # original
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu") # changed:change to GPU
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
