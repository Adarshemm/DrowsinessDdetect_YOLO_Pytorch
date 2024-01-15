import torch                        #used to load YOLO model and make decisions
from matplotlib import pyplot as plt #used to visualize image
import numpy as np                  #used for array transformation
import cv2 
import uuid
import os
import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

img = os.path.join('data','images','awake.bbb0ecfc-b25c-11ee-8298-bf4288770f37.jpg')
results = model(img)

results.print()