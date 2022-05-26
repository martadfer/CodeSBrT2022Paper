import cv2
import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython.display as ipd
import numpy as np
import pandas as pd
from IPython.display import Audio
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from tensorflow.keras import applications
from tensorflow.keras.layers import Input
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json

from tqdm import tqdm_notebook


def map_keras_label(x):
    if(x == 0):
        # Negativos são 0
        return 'negativo'
    elif(x == 1):
        # Neutros são 1
        return 'neutro'
    elif(x==2):
        # Positivos são 2
        return 'positivo'
    else:
        # Surprise é 3
        return 'surpresa'

# Modelo
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

dlib.DLIB_USE_CUDA = True
size = 100

model = loaded_model

print("Modelo Carregado")

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

captura = cv2.VideoCapture(0)
 
count = 0
count_photo = 0

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

while(1):
    ret, frame = captura.read()


    if(count%1==0):
        rects = detector(frame, 0)

        #For each detected face  
        for k,d in enumerate(rects):

            (x, y, w, h) = rect_to_bb(d)

            if((count_photo%20 == 0)|(count_photo == 0)):
                face = frame[y - 50 :y + h + 30, x - 20:x + w + 20]
                # video = plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face = cv2.resize(cv2.cvtColor(face , cv2.COLOR_BGR2RGB), (size,size))
                face = np.expand_dims(np.array(face, dtype=np.float), axis=0)
                preds = model.predict(face)
                print(preds[0])
                print(map_keras_label(np.argmax(preds[0])))


            cv2.rectangle(frame, (x - 20, y - 50), (x + 20 + w, y + 20 +  h), (0, 255, 0), 2)
            cv2.putText(frame, map_keras_label(np.argmax(preds[0])), (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2, cv2.LINE_AA)
            #Get coordinates
            shape = predictor(frame, d)
            #There are 68 landmark points on each face
            # for i in range(1,68):

            #     cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=-1) #For each point, draw a red circle with thickness2 on the original frame


    
        cv2.imshow("Video", frame)
    

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count = count+1
    count_photo = count_photo + 1


captura.release()
cv2.destroyAllWindows()