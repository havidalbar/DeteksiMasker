import cv2
from model import FacialMaskModel
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# facec = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialMaskModel("model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE) 

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            fc = cv2.cvtColor(fc, cv2.COLOR_BGR2RGB)
            fc = cv2.resize(fc, (224, 224))
            fc = img_to_array(fc)
            fc = np.expand_dims(fc, axis=0)
            fc =  preprocess_input(fc)
            if len(fc)>0:
                pred = model.predict_mask(fc)
                
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
