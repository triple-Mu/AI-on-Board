import cv2
import numpy as np
from rknn.api import RKNN
from numpy import ndarray
from typing import List
from dataclasses import dataclass

ONNX_MODEL = 'yolov5s.onnx'
RKNN_MODEL = 'yolov5s.rknn'
IMG_PATH = 'bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow",
           "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut",
           "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ",
           "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ",
           "hair drier", "toothbrush ")

ANCHORS = [[1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
           [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375],
           [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]]


@dataclass
class Object:
    label: int
    score: float
    box: ndarray


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def yolov5_decode(feats: List[ndarray],
                  anchors: List[List[float]],
                  conf_thres: float):
    proposals: List[Object] = []
    for i, feat in enumerate(feats):
        stride = 8 << i
        feat_h, feat_w = feat.shape[1:-1]
        anchor = anchors[i]
        feat = sigmoid(feat)
        feat = feat.reshape((feat_h, feat_w, 3, -1))
        box_feat, conf_feat, score_feat = np.split(feat, [4, 5], -1)

        hIdx, wIdx, aIdx, _ = np.where(conf_feat > conf_thres)

        num_proposal = hIdx.size
        if not num_proposal:
            continue

        score_feat = score_feat[hIdx, wIdx, aIdx] * conf_feat[hIdx, wIdx, aIdx]
        boxes = box_feat[hIdx, wIdx, aIdx]
        labels = score_feat.argmax(-1)
        scores = score_feat.max(-1)

        indices = np.where(scores > conf_thres)[0]
        if len(indices) == 0:
            continue

        for idx in indices:
            a_w = anchor[2 * aIdx[idx]]
            a_h = anchor[2 * aIdx[idx] + 1]
            x, y, w, h = boxes[idx]
            x = (x * 2.0 - 0.5 + wIdx[idx]) * stride
            y = (y * 2.0 - 0.5 + hIdx[idx]) * stride
            w = (w * 2.0) ** 2 * a_w * stride
            h = (h * 2.0) ** 2 * a_h * stride

            x1 = x - w / 2
            y1 = y - h / 2

            proposals.append(Object(labels[idx], scores[idx], np.array([x1, y1, w, h], dtype=np.float32)))
    return proposals


def nms(proposals: List[Object], conf_thres: float = 0.25, iou_thres: float = 0.65):
    bboxes = []
    scores = []
    class_ids = []
    for proposal in proposals:
        bboxes.append(proposal.box)
        scores.append(proposal.score)
        class_ids.append(proposal.label)
    indices = cv2.dnn.NMSBoxesBatched(bboxes, scores, class_ids, conf_thres, iou_thres)
    results = []
    for idx in indices:
        result = proposals[idx]
        result.box[2:] += result.box[:2]
        results.append(result)
    return results


def draw(image, results: List[Object]):
    for result in results:
        box = result.box
        cl = result.label
        score = result.score
        top, left, right, bottom = box
        print(f'class: {CLASSES[cl]}, score: {score}')
        print(f'box coordinate left,top,right,down: [{top}, {left}, {right}, {bottom}]')
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}',
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588s')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    # post process
    proposals = yolov5_decode(outputs, ANCHORS, OBJ_THRESH)
    results = nms(proposals, OBJ_THRESH, NMS_THRESH)

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(results) > 0:
        draw(img_1, results)
        cv2.imwrite('result.jpg', img_1)

    rknn.release()
