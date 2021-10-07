# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
#Solomon Ubani ( solomonubani@my.unt.edu )
#Sulav Poudyal ( sulav697@gmail.com )
#Yen Pham ( yenpham@my.unt.edu )
#Khoa Ho ( khoaho@my.unt.edu ) 
#Stephanie Brooks( StephanieBrooks2@my.unt.edu )

# import libraries/packages
import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from gtts import gTTS
import pyttsx3
import threading





# initialize text to speech engine
tTSEngine = pyttsx3.init()

# language for converting text to audio
speechLanguage = 'en_US'
voiceGender = 'voiceGenderFemale'
speechSpeedRate = 125

tTSEngine.setProperty('voice', tTSEngine.getProperty('voices')[2].id)
tTSEngine.setProperty('rate', speechSpeedRate)

#for voice in tTSEngine.getProperty('voices'):
#    if speechLanguage in voice.languages and voiceGender == voice.gender:
#        tTSEngine.setProperty('voice', voice.id)
#        break

def produceSpeech (tTSEngine, text):
    tTSEngine.say(text)
    tTSEngine.runAndWait()
    tTSEngine.stop()


class speechThread (threading.Thread):
    def __init__(self, tTSEngine, text):
        threading.Thread.__init__(self)
        self.tTSEngine = tTSEngine
        self.text = text
    def run(self):
        produceSpeech(self.tTSEngine, self.text)


speechToggle = False

def toggleState (currentState):
    return not currentState

class toggleThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.currentState = False
    def run(self):
        while True:
            if cv2.waitKey(1) == ord('s'):
                self.currentState = toggleState(self.currentState)

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

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, static_image_mode=False)

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
newModel = load_model("newMPHandGestureModel_24classes")

# Load class names
#gestureFile = open('gesture.names', 'r')
#classNames = gestureFile.read().split('\n')
#gestureFile.close()

newGFile = open("gestureNames24classes.txt", 'r')
classNames = newGFile.read().split('\n')
classNames.append("unknown")
# print(classNames)
# print(classNames[-1])

# Initialize the webcam for Hand Gesture Recognition Python project
cameraPort = 0
cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)


#import inspect

#sig = inspect.signature(model.fit)
#print(str(sig))

# keep running & capturing video/images
#while True:
# while camera is running:
while cap.isOpened():
    # Read each frame from the webcam
    capSuccess, frame = cap.read()
    x , y, c = frame.shape
    # Flip the frame vertically
    #frame = cv2.flip(frame, 1)
    
    # detect key points
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    handResult = hands.process(framergb)
    className = ''
    
    # faceResult = faceDetection.process(framergb)
    faceResult = faceMesh.process(framergb)

    # post process the result
    # for hands 
    if handResult.multi_hand_landmarks:
        # print("hand result: ", handResult.multi_hand_landmarks)
        multiLandmarks = []
        landmarks = []
        for handslms in handResult.multi_hand_landmarks:
            # print(handslms)
            landmarks = []
            for lm in handslms.landmark:
                # print(lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                # lmz = int(lm.z)
                landmarks.append([lmx, lmy])
            # end inner for-loop

            # print(landmarks)
            # Drawing landmarks on frames
            mpDrawing.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())

            # Predict gesture in Hand Gesture Recognition project
            # note: landmarks = [ 21x[] ] -> wrapped -> [landmarks] = [ [ 21x[] ] ]
            # the model can accept [ landmarks , landmarks, ... , landmarks ] = [ [ 21x[] ], [ 21x[] ], ... , [ 21x[] ] ]
            # it is meant to accept "a list containing multiple lists of landmarks",
            # in which case it will also output a "list of lists of probabilities"

            #prediction = model.predict([landmarks])
            prediction = newModel.predict([landmarks])
            
            # prediction output is in form [ [probability_1, probability_2, ... , probability_n] ] with n = number of classes
            # print(prediction)
            
            #classIndices = []
            #classProbabilities = []
            #for idx, classProbability in enumerate(prediction[0]):
                # print(classProbability)
            #    if classProbability > 0.8 :
            #        classIndices.append(idx)
            #        classProbabilities.append(classProbability)

            #if len(classIndices) == 0:
            #   print("unknown class")
            #    className = classNames[-1]
            #elif len(classIndices) == 1:
            #    print("1 class")
            #    classIndex = classIndices[0]
            #    className = classNames[classIndex]
            #else:
            #    print("multi classes")
            #    classIndex = classIndices[ np.argmax(classProbabilities) ]
            #    className = classNames[classIndex]
                
            # making decision on which class of gesture the captured frame should be
            classIndex = np.argmax(prediction[0])
            # print("class index: ", classIndex)
            
            if prediction[0][classIndex] > 0.7 :
                className = classNames[classIndex]
                print("1 class >70% prob: ", className)
            else:
                print("unsure class")
                className = classNames[-1]
        # end outer for-loop
        

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        # show frame rate per second
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame, "fps: " + str(int(fps)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)   
    
    # end if for hands

    # for face
    #if faceResult.multi_face_landmarks:
        #for facelms in faceResult.multi_face_landmarks:
            # for lm in facelms.landmark:
                # print(lm)
                # lmx = int(lm.x * x)
                # lmy = int(lm.y * y)
                # lmz = int(lm.z)
                # faceLandmarks.append([lmx, lmy])
            # end for-loop

            ##mpDrawing.draw_detection(frame, detectedFace)
            #mpDrawing.draw_landmarks(image=frame, landmark_list=facelms, connections=mpFaceMesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style())
            #mpDrawing.draw_landmarks(image=frame, landmark_list=facelms, connections=mpFaceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())
        # end for-loop
    # end if for face


    # Show the final output
    cv2.imshow("Output", frame)
    
    threadNotExist = True
    try:
        spThread
        threadNotExist = False
    except NameError:
        threadNotExist = True
    else:
        threadNotExist = False

    # convert to speech # would block/slow down the app
    if speechToggle and className !="" and className != "unknown" and (threadNotExist or spThread.is_alive() == False) :
        spThread = speechThread(tTSEngine, className)
        spThread.start()
        #spThread.join()
        #tTSEngine.say(className)
        #tTSEngine.runAndWait()
        # stop the speech engine
        #tTSEngine.stop()
    
    # toggle speech on/off when 's' key is pressed
    if cv2.waitKey(1) == ord('s'):
        speechToggle = toggleState(speechToggle)
    # stop the app when the 'q' key is pressed
    elif cv2.waitKey(1) == ord('q'):
        break



# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()



    