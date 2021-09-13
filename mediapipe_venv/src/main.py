
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



# initialize time mark
cTime = 0
pTime = 0

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mpDrawing = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles


# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
gestureFile = open('gesture.names', 'r')
classNames = gestureFile.read().split('\n')
gestureFile.close()


# Initialize the webcam for Hand Gesture Recognition Python project
cameraPort = 0
cap = cv2.VideoCapture(cameraPort, cv2.CAP_DSHOW)

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



# keep running & capturing video/images
while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x , y, c = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    
    # detect key points
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mpDrawing.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS, mpDrawingStyles.get_default_hand_landmarks_style(), mpDrawingStyles.get_default_hand_connections_style())

            #print(landmarks)
            #print([landmarks])
            # Predict gesture in Hand Gesture Recognition project
            prediction = model.predict([landmarks])
            #print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
        
        
        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        # show frame rate per second
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(frame, "fps: " + str(int(fps)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)   
    
    # Show the final output
    cv2.imshow("Output", frame)
    
    # convert to speech # would block/slow down the app
    tTSEngine.say(className)
    tTSEngine.runAndWait()

    # stop the app when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# stop the speech engine
tTSEngine.stop()

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()



    