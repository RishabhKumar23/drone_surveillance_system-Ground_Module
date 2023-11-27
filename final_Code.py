import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
from gtts import gTTS
import pyttsx3
import zmq

# Function to resize images
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# Load images and names from the directory
path = "authorization_persons"
studentImg = []
studentName = []
myList = os.listdir(path)

for cl in myList:
    curimg = cv2.imread(f"{path}/{cl}")
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

# Function to find face encodings
def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

# Function to mark attendance
def markAttendance(name):
    with open("Timing.csv", "r+") as f:
        myDatalist = f.readlines()
        nameList = [entry.split(",")[0] for entry in myDatalist]

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime("%H:%M")
            f.writelines(f"\n{name}, {timestr}")
            statement = f"Welcome, {name}"
            tts = gTTS(text=statement, lang="en")
            tts.save("welcome.mp3")
            os.system("mpg321 welcome.mp3")  # install mpg321

# Find face encodings for loaded images
EncodeList = findEncoding(studentImg)

# Using ZeroMQ
# context = zmq.Context()

# Create REQ sockets for two cameras
# socket1 = context.socket(zmq.REQ)
# socket1.connect("tcp://192.168.149.85:5555")  # Connect to the publisher's address

# Open two video capture objects for two cameras
vid1 = cv2.VideoCapture(0)
vid2 = cv2.VideoCapture(2)

while True:
    # Read frames from both cameras
    success1, frame1 = vid1.read() 
    success2, frame2 = vid2.read()

    # Check if frames are successfully captured
    if not success1:
        print("Error capturing frame from Camera 1")
        break

    if not success2:
        print("Error capturing frame from Camera 2")
        break

    # Resize frames
    smaller_frame1 = cv2.resize(frame1, (0, 0), None, 0.25, 0.25)
    smaller_frame2 = cv2.resize(frame2, (0, 0), None, 0.25, 0.25)

    # Find faces in the frames
    faces_in_frame1 = face_rec.face_locations(smaller_frame1)
    encode_faces_in_frame1 = face_rec.face_encodings(smaller_frame1, faces_in_frame1)

    faces_in_frame2 = face_rec.face_locations(smaller_frame2)
    encode_faces_in_frame2 = face_rec.face_encodings(smaller_frame2, faces_in_frame2)

    # Iterate over faces in camera 1
    for encode_face, face_loc in zip(encode_faces_in_frame1, faces_in_frame1):
        matches = face_rec.compare_faces(EncodeList, encode_face)
        face_dis = face_rec.face_distance(EncodeList, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = studentName[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame1, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame1,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            markAttendance(name)
            message = "Authorize Person"
            print(f"Published (Camera 1): {message}")

    # Iterate over faces in camera 2
    for encode_face, face_loc in zip(encode_faces_in_frame2, faces_in_frame2):
        matches = face_rec.compare_faces(EncodeList, encode_face)
        face_dis = face_rec.face_distance(EncodeList, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = studentName[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame2, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame2,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            markAttendance(name)
            message = "Authorize Person"
            print(f"Published (Camera 2): {message}")

    # Display frames from both cameras
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid1.release()
vid2.release()
cv2.destroyAllWindows()
