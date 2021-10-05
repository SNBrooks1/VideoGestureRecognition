# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
#Solomon Ubani ( solomonubani@my.unt.edu )
#Sulav Poudyal ( sulav697@gmail.com )
#Yen Pham ( yenpham@my.unt.edu )
#Khoa Ho ( khoaho@my.unt.edu ) 
#Stephanie Brooks( StephanieBrooks2@my.unt.edu )

# module to train a classifer-model -> current choice: LSTM

# import libraries/packages
import os
import time
import math
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as pyplt
#import pandas

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

#from google.colab.patches import cv2_imshow
#from keras.preprocessing import image
#from keras.utils import np_utils

import threading
import json


# initialize time mark
cTime = 0
pTime = 0

# initialize mediapipe
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7)

# mpFaceDetection = mp.solutions.face_detection
# faceDetection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.8)
## model_selection: 0 -> within 2 meters from camera, 1 -> 2-5 meters from camera

#mpFaceMesh = mp.solutions.face_mesh
#faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, static_image_mode=False)

# Load class names


newGFile = open("new_gesture_names.txt", 'r')
newGClassNames = newGFile.read().split('\n')
numOfGClasses = len(newGClassNames)
#print(newGClassNames)
#print(numOfGClasses)

# load data
dataFilePath = r"./data_handLms.txt"
inputFile = open(dataFilePath, "r")

readLines = inputFile.readlines()
inputFile.close()

xsTrain = []
ysTrain = []

for readLine in readLines:
    dInstance = json.JSONDecoder().decode(readLine)
    #print(dInstance)
    handLandmarks = dInstance["landmarks"]
    gestureClass = dInstance["class"]
    #print(gestureClass)
    gClassId = newGClassNames.index(gestureClass)
    gClassProbList = [0] * numOfGClasses
    gClassProbList[gClassId] = 1
    #print("id: ", gClassId)
    xsTrain.append( [[handLandmarks]] )
    ysTrain.append(gClassProbList)
    


# Training tracking
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Load the gesture recognizer model
model = load_model("mp_hand_gesture")

# Initiate a new model
newModel = Sequential()

# Copying all the layers from the old model, except the output layers
for layer in model.layers[:-1]: # just exclude last layer from copying
    newModel.add(layer)

# if necessary, prevent the already trained layers from being trained again 
# for layer in newModel.layers:
    #layer.trainable = False

# adding the new output layer, 
# note: the "name" parameter is important and should be unique here to avoid error
newModel.add(Dense(numOfGClasses, name='newDenseLayer', activation='softmax'))

# compile the new Model
newModel.compile(optimizer='Adam',loss='categorical_crossentropy', metrics = ['categorical_accuracy'])

# train
newModel.fit(x=xsTrain, y=ysTrain, epochs=10, callbacks=[tb_callback])

# save it
newModel.save("new_mp_hand_gesture_model")







