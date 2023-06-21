import argparse
import cv2
import time
import random
import numpy as np
from pathlib import Path
from queue import Queue
from rknnlite.api import RKNNLite  # noqa
from triplemu import postprocess_yolov8  # noqa
from concurrent.futures import ThreadPoolExecutor

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

NUM_CLASSES = 80


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Image path, images path or video path.')
    parser.add_argument('--rknn', type=str, help='Rknn path file')
    parser.add_argument('--output', type=str, default='output', help='Output path for saving results.')
    parser.add_argument('--num_thread', type=int, default=3, help='Num threads')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument('--num_classes', type=int, default=80, help='Num classes')
    args = parser.parse_args()
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes
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


def run(rknn_lite: RKNNLite, image_bgr: np.ndarray,
        conf_thres: float = 0.25, iou_thres: float = 0.65,
        net_h: int = 640, net_w: int = 640, num_classes: int = NUM_CLASSES) -> np.ndarray:
    orin_h, orin_w = image_bgr.shape[:2]
    ratio_h = net_h / orin_h
    ratio_w = net_w / orin_w
    _image_bgr = cv2.resize(image_bgr, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(_image_bgr, cv2.COLOR_BGR2RGB)
    outputs = rknn_lite.inference(inputs=[image_rgb])
    outputs = postprocess_yolov8(*outputs,
                                 conf_thres=conf_thres, iou_thres=iou_thres,
                                 net_h=net_h, net_w=net_w,
                                 orin_h=orin_h, orin_w=orin_w,
                                 ratio_h=ratio_h, ratio_w=ratio_w,
                                 num_classes=num_classes)
    draw_on_image(image_bgr, outputs)
    return image_bgr


class rknnPoolExecutor:
    def __init__(self, rknn_path, num_thread, func):
        assert num_thread > 0
        self.num_thread = num_thread
        self.queue = Queue()
        self.rknnPool = [load_rknn(rknn_path, i % 3) for i in range(num_thread)]
        self.pool = ThreadPoolExecutor(max_workers=num_thread)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(
            self.pool.submit(
                self.func, self.rknnPool[self.num % self.num_thread], frame
            ))
        self.num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()


def main():
    args = parse_args()
    inputs = Path(args.input)
    output_path = Path(args.output)
    if not args.show:
        output_path.mkdir(parents=True, exist_ok=True)

    assert inputs.suffix == '.mp4'
    pools = rknnPoolExecutor(args.rknn, args.num_thread, run)

    for i in range(args.num_thread + 1):
        pools.put(np.zeros((640, 640, 3), dtype=np.uint8))

    cap = cv2.VideoCapture(str(inputs))
    frame_id = 0

    loopTime, initTime = time.perf_counter(), time.perf_counter()
    fps = 0
    now = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        save_path = output_path / f'frame_{frame_id:04d}.jpg'
        pools.put(frame)
        image_bgr, flag = pools.get()
        if not flag:
            break
        if args.show:
            cv2.imshow('result', image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.imwrite(str(save_path), image_bgr)

        if pools.num % 30 == 0:
            now = time.perf_counter()
            fps = 30 / (now - loopTime)
            print(f'30帧平均帧率: {fps:0>6.2f}帧')
            loopTime = now
    now = time.perf_counter()
    fps = pools.num / (now - initTime)
    print(f'总平均帧率: {fps:0>6.2f}帧')
    cap.release()
    pools.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
