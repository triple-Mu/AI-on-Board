# YOLOv5 deploy in RK3588

## 1. Prepare your environment

### 1.1 On X86 PC

Suggest to use anaconda to create a virtual environment.

```bash
conda create -n rknn python=3.8
conda activate rknn
```

Install yolov8:

```bash
git clone https://github.com/triple-Mu/yolov5.git -b triplemu/model-only
# install yolov5
cd yolov5
pip install -r requirements.txt
```

Convert pt(yolov5s.pt) to onnx(yolov5s.onnx):

```bash
python export.py --weights yolov5s.pt --include onnx --simplify
```

Convert onnx to rknn:

`rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl` is in `packages`

```bash
cd AI-on-Board/Rockchip/cpp/yolov5/pysrc
pip install rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl
# modify the onnx2rknn.py: ONNX_MODEL RKNN_MODEL IMG_PATH DATASET IMG_SIZE
# get yolov5s.rknn model
sh run.sh
```

### 1.2 On ARM RK3588

Copy this repo to your board and build rknn-yolov5 demo(single thread and multithread).

```bash
cd AI-on-Board/Rockchip/cpp/yolov5
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## 2. Run

```bash
./rknn-yolov5 ./pysrc/yolov5s.rknn ./test.mp4
./rknn-yolov5-mt ./pysrc/yolov5s.rknn ./test.mp4
```

`test.mp4` is your own mp4 path.
