#!/bin/bash
echo "Now running on your x86 PC!"
find ${PWD}/coco128/ -type f | xargs -i readlink -f {} >dataset.txt
python3 onnx2rknn.py