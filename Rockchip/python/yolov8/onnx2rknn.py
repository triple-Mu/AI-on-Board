import cv2
from rknn.api import RKNN

ONNX_MODEL = 'yolov8s.onnx'
RKNN_MODEL = 'yolov8s.rknn'
IMG_PATH = 'zidane.jpg'
DATASET = 'imagelist.txt'
IMG_SIZE = 640

QUANTIZE_ON = True

# Create RKNN object
rknn = RKNN(verbose=True)

# pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            quantized_algorithm='kl_divergence',
            quantized_method='layer',
            target_platform='rk3588s',
            custom_string='yolov8s')
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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Inference
print('--> Running model')
outputs = rknn.inference(inputs=[img])
print('done')

rknn.release()
