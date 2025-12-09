from torchinfo import summary
from ultralytics import YOLO
import torch

model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml').model

model.info(detailed=True)
print(model.info(detailed=True))