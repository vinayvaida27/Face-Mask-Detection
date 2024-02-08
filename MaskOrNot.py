# -*- coding: utf-8 -*-
"""
@author: Vinay Vaida
"""

# Importing the libraries
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\vinay\Jupyter Notebook\Face-mask-detection\maskDetectionModel.h5')

face_cascade = cv2.CascadeClassifier(r'C:\Users\vinay\Jupyter Notebook\Face-mask-detection\haarcascade_frontalface_default.xml')

def face_extractor(img):
    scale_factor = 1.05
    min_neighbour = 6
    faces = face_cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbour, minSize=(100, 100),
                                           flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        return None
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()  
    face = face_extractor(frame)
    
    if isinstance(face, np.ndarray):
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        
        if pred[:, 1] > 0.001:
            name = 'No mask found'
        else:
            name = 'Mask found'
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
