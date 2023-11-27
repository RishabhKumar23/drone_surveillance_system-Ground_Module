import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
from gtts import gTTS
import pyttsx3
import zmq

# vid1 = cv2.VideoCapture(0)
vid2 = cv2.VideoCapture(2)

while True:
    # success1, frame1 = vid1.read()
    success2, frame2 = vid2.read()

    # cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# vid1.release()
vid2.release()
cv2.destroyAllWindows()
