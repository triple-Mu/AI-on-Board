# YOLOv8 deploy in RK3588

## 1. Prepare your environment

### 1.1 On X86 PC

Suggest to use anaconda to create a virtual environment.

```bash
conda create -n rknn python=3.8
conda activate rknn
```

Install yolov8:

```bash
git clone https://github.com/triple-Mu/yolov8.git -b triplemu/model-only
# uninstall ultralytics first
pip uninstall ultralytics
# install yolov8
cd yolov8
pip install -r requirements.txt
pip install .
```

Convert pt to onnx:

```bash
git clone https://github.com/triple-Mu/AI-on-Board.git
cd AI-on-Board/Rockchip/python/yolov8
# modify the export.py: pt_path to your own first
python export.py
```

Convert onnx to rknn:

`rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl` is in `packages`

```bash
pip install rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl
# modify the onnx2rknn.py: ONNX_MODEL RKNN_MODEL IMG_PATH DATASET IMG_SIZE
python onnx2rknn.py
```

### 1.2 On ARM RK3588

Copy this repo to your board.

Install rknn-lite and triplemu tools:
`rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl` and `triplemu-0.0.1-cp38-cp38-linux_aarch64.whl` is in `packages`

```bash
cd AI-on-Board/Rockchip/python/yolov8
# install rknn_toolkit_lite and triplemu tools on RK3588
pip install rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl
pip install triplemu-0.0.1-cp38-cp38-linux_aarch64.whl
```

## 2. Run

```bash
python rknn_infer.py --input zidane.jpg --rknn yolov8s.rknn --show
```

### Description of all arguments

- `--input` : The image path or images dir or mp4 path.
- `--rknn` : The rknn model path.
- `--show` : Whether to show results.
- `--output` : The output dir path for saving results.
- `--iou-thres` : IoU threshold for NMS algorithm.
- `--conf-thres` : Confidence threshold for NMS algorithm.