import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
from gtts import gTTS
import os

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

path = 'authorization_persons'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

def markAttendance(name):
    with open('Timing.csv', 'r+') as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statement = f'Welcome to class, {name}'
            tts = gTTS(text=statement, lang='en')
            tts.save('welcome.mp3')
            os.system('mpg321 welcome.mp3')  # install mpg321

EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(0)
while True:
    success, frame = vid.read()
    smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    faces_in_frame = face_rec.face_locations(smaller_frames)
    encode_faces_in_frame = face_rec.face_encodings(smaller_frames, faces_in_frame)

    for encode_face, face_loc in zip(encode_faces_in_frame, faces_in_frame):
        matches = face_rec.compare_faces(EncodeList, encode_face)
        face_dis = face_rec.face_distance(EncodeList, encode_face)
        print(face_dis)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = studentName[match_index].upper()
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
