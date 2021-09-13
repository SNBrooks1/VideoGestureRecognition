# University of North Texas
# Fall 2021
# Team project for class CSCE 5280 by Professor Mark Albert

# Team members:
Solomon Ubani ( solomonubani@my.unt.edu )
Sulav Poudyal ( sulav697@gmail.com )
Yen Pham ( yenpham@my.unt.edu )
Khoa Ho ( khoaho@my.unt.edu ) 
Stephanie Brooks( StephanieBrooks2@my.unt.edu )


# Video Hand Gesture & Facial Expression Recognition using Key-Points Detection
Use MediaPipe libraries to detect landmarks (key-points) from a real-time video
Use Tensorflow models that can recognize a certain set of poses, gestures, and expression through input set of landmarks.
Combine the predicted pose, gesture, and facial expression to result in a verbal meaning from a user-defined dictionary (that can be redefined/adjusted as necessary)
Use text-to-speech engine to produce audio output of the verbal terms.


# To run:
0/ Requirements: install the following python packages
venv
numpy
cv2
mediapipe
tensorflow
pyttsx3
gtts

1/ Setup the "mediapipe_venv" directory as a virtual environment:
* Find documentations on virtual environment here: https://docs.python.org/3/tutorial/venv.html
** On a shell console, relocate to the directory where you clone/store the repository on your computer, use command:  python -m venv mediapipe_venv
*** Actual command may differ based on OS, your current directory, whether you are using a shell console or python intepreter, or is already in a virtual environment (managed through anaconda, jupyter, or any other module)

2/ Activate python virtual environment, using OS-specific script in "mediapipe_venv\scripts"


3/ Application main entry: "mediapipe_venv\src\main.py"
* Execute it with python, i.e., move to the said directory ("mediapipe_venv\src"), then use command "python main.py" (or "python3 main.py", depending on your computer setup)

# References:
https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/
https://google.github.io/mediapipe/solutions/hands.html
https://google.github.io/mediapipe/getting_started/python.html
