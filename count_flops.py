from ultralytics import YOLO
from ptflops import get_model_complexity_info
import torch

# load model
# yolo = YOLO('yolov8n')
yolo = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

# get the real torch model
net = yolo.model
net.eval()

# # choose resolution
# input_res = (3,640,640)

# # compute
# flops, params = get_model_complexity_info(net, input_res, as_strings=True)

# print("FLOPs:", flops)
# print("Parameters:", params)

input_res = (3,320,320)

# compute
flops, params = get_model_complexity_info(net, input_res, as_strings=True)

print("FLOPs:", flops)
print("Parameters:", params)
