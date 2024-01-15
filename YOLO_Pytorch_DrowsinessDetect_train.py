#Install pytorch -> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#Clone the YOLOv5 from the git and the install the dependcies
# git clone https://github.com/ultralytics/yolov5  # clone
# cd yolov5
# pip install -r requirements.txt  # install

import torch                        #used to load YOLO model and make decisions
from matplotlib import pyplot as plt #used to visualize image
import numpy as np                  #used for array transformation
import cv2                          #helps to access webcam

#load the model from the YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#make detections with online images
img = 'https://th.bing.com/th/id/OIP.VWMpJs65EFzeywcI-LeZsgHaFY?w=262&h=190&c=7&r=0&o=5&dpr=1.5&pid=1.7' #'https://ultralytics.com/images/zidane.jpg' #from the coco class -> https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVZZOVdCaDEzVmpOaFYya1M5N1l2ODBQckVGUXxBQ3Jtc0trZElxRVBzOVZrY0ZHQjZOVjhiWlU4bHRUdXpBUkd4ZXFvZTQwd29xbmI1Y05RQTZTb3lwVlljblM5X1ZhUldfbTFzcXdHZ3pOVjhLd2I2Nmt3N0lCbUd5d2c3Ump0NHZENTMxZmRSWGFoMk5IckFRaw&q=https%3A%2F%2Fgist.github.com%2FAruniRC%2F7b3dadd004da04c80198557db5da4bda&v=tFNJGim3FXw

results = model(img)
results.print()

# %matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.show()

results.show()
results.render()

###################################################################
#   REAL TIME using OpenCV

cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()

#     #make detections
#     results = model(frame)

#     cv2.imshow('YOLO', np.squeeze(results.render()))

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import uuid
import os
import time

IMAGE_PATH = os.path.join('data', 'images') #/data/images
labels = ['awake', 'drowsy']
number_imgs = 20

cap = cv2.VideoCapture(0)
#loop through labels
for label in labels:
    print('Collecting images for {}'.format(labels))
    time.sleep(5)

    #loop through image range 20
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        #webcam feed
        ret, frame = cap.read()

        imgname = os.path.join(IMAGE_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        
        #render to the screen
        cv2.imshow('image collection', frame)
        time.sleep(2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#install -> pip install pyqt5 lxml --upgrade
#and after that -> cd labelImg 
#and to open the labelling resource -> pyrcc5 -o libs/resources.py resources.qrc

####################
# now to train the model 
# cd .\yolov5\
#and then -> python train.py --img 320 --batch 16 --epochs 500 --data dataset.yml --weights yolov5s.pt --workers 2 
