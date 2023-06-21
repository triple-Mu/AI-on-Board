import argparse
import cv2
import random
import numpy as np
from pathlib import Path
from rknnlite.api import RKNNLite  # noqa
from triplemu import postprocess_yolov8  # noqa

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

random.seed(0)

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Image path, images path or video path.')
    parser.add_argument('--rknn', type=str, help='Rknn path file')
    parser.add_argument('--output', type=str, default='output', help='Output path for saving results.')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.25, help='Bbox score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.65, help='Bbox iou threshold')
    args = parser.parse_args()
    return args


def load_rknn(rknn_path: str, core_id: int = 0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknn_path)
    assert ret == 0

    if core_id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif core_id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif core_id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif core_id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    assert ret == 0
    return rknn_lite


def run(rknn_lite: RKNNLite, image_bgr: np.ndarray,
        conf_thres: float, iou_thres: float,
        net_h: int, net_w: int, num_classes: int = 80) -> list:
    orin_h, orin_w = image_bgr.shape[:2]
    ratio_h = net_h / orin_h
    ratio_w = net_w / orin_w
    image_bgr = cv2.resize(image_bgr, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    outputs = rknn_lite.inference(inputs=[image_rgb])
    outputs = postprocess_yolov8(*outputs,
                                 conf_thres=conf_thres, iou_thres=iou_thres,
                                 net_h=net_h, net_w=net_w,
                                 orin_h=orin_h, orin_w=orin_w,
                                 ratio_h=ratio_h, ratio_w=ratio_w,
                                 num_classes=num_classes)
    return outputs


def draw_on_image(image_bgr: np.ndarray, outputs: list):
    for output in outputs:
        x1 = output.x1
        y1 = output.y1
        x2 = output.x2
        y2 = output.y2
        cls = CLASSES[output.label]
        color = COLORS[cls]
        cv2.rectangle(image_bgr, [x1, y1], [x2, y2], color, 2)
        cv2.putText(image_bgr,
                    f'{cls}:{output.score:.3f}', (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [0, 0, 225],
                    thickness=2)


def main():
    args = parse_args()
    rknn_lite = load_rknn(args.rknn)
    inputs = Path(args.input)
    output_path = Path(args.output)
    if not args.show:
        output_path.mkdir(parents=True, exist_ok=True)

    if inputs.suffix in IMG_EXTENSIONS:
        image = cv2.imread(str(inputs))
        save_path = output_path / inputs.name
        outputs = run(rknn_lite, image, args.score_thr, args.iou_thr, 640, 640, num_classes=len(CLASSES))
        draw_on_image(image, outputs)
        if args.show:
            cv2.imshow('result', image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_path), image)
    elif inputs.is_dir():
        for image_path in inputs.iterdir():
            if image_path.suffix in IMG_EXTENSIONS:
                image = cv2.imread(str(image_path))
                save_path = output_path / inputs.name
                outputs = run(rknn_lite, image, args.score_thr, args.iou_thr, 640, 640, num_classes=len(CLASSES))
                draw_on_image(image, outputs)
                if args.show:
                    cv2.imshow('result', image)
                    cv2.waitKey(0)
                else:
                    cv2.imwrite(str(save_path), image)
    elif inputs.suffix == '.mp4':
        cap = cv2.VideoCapture(str(inputs))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            save_path = output_path / f'frame_{frame_id:04d}.jpg'
            outputs = run(rknn_lite, frame, args.score_thr, args.iou_thr, 640, 640, num_classes=len(CLASSES))
            draw_on_image(frame, outputs)
            if args.show:
                cv2.imshow('result', frame)
                if cv2.waitKey(1) == 27:
                    break
            else:
                cv2.imwrite(str(save_path), frame)


if __name__ == '__main__':
    main()
