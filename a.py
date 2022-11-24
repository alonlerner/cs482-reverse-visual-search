from pycocotools.coco import COCO
import torch
import torchvision
from torchvision import datasets, io, models, ops, transforms, utils

ann_file = "./instances_minitrain2017.json"
coco=COCO(ann_file)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()