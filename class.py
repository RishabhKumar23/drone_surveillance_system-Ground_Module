import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from gtts import gTTS
import pyttsx3
import threading
import socket

# connection to server
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client_socket.connect(("192.168.221.85",8889)) # server ip and port 

class FaceRecognitionSystem:
    def __init__(self, path="authorization_persons", video_source=0):
        self.path = path
        self.video_source = video_source
        self.studentImg = []
        self.studentName = []
        self.EncodeList = []

        # Load images and names from the directory
        self.load_images()

    def load_images(self):
        myList = os.listdir(self.path)

        for cl in myList:
            cur_img = cv2.imread(f"{self.path}/{cl}")
            self.studentImg.append(cur_img)
            self.studentName.append(os.path.splitext(cl)[0])

        # Find face encodings for loaded images
        self.EncodeList = self.find_encoding(self.studentImg)

    @staticmethod
    def resize(img, size):
        width = int(img.shape[1] * size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

    def find_encoding(self, images):
        img_encodings = []
        for img in images:
            img = self.resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode_img = face_recognition.face_encodings(img)[0]
            img_encodings.append(encode_img)
        return img_encodings

    def mark_attendance(self, name):
        with open("Timing.csv", "r+") as f:
            my_data_list = f.readlines()
            name_list = [entry.split(",")[0] for entry in my_data_list]

            if name not in name_list:
                now = datetime.now()
                time_str = now.strftime("%H:%M")
                f.writelines(f"\n{name}, {time_str}")
                statement = f"Welcome, {name}"
                tts = gTTS(text=statement, lang="en")
                tts.save("welcome.mp3")
                os.system("mpg321 welcome.mp3")  # install mpg321

    def face_dec(self, frame, cam):
        # Resize frames
        smaller_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

        # Find faces in the frames
        faces_in_frame = face_recognition.face_locations(smaller_frame)
        encode_faces_in_frame = face_recognition.face_encodings(smaller_frame, faces_in_frame)

        # Iterate over faces in the camera
        for encode_face, face_loc in zip(encode_faces_in_frame, faces_in_frame):
            matches = face_recognition.compare_faces(self.EncodeList, encode_face)
            face_dis = face_recognition.face_distance(self.EncodeList, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = self.studentName[match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    name,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                self.mark_attendance(name)
                message = "Authorize Person"
                print(f"Published (Camera {cam}): {message}")
            else:
                print(f"Unauthorize Person at: {cam}")
                unAuth = f"Unauthorize Person at: {cam}"
                print(unAuth)
                client_socket.send(unAuth.encode())
        return frame

    def run_system(self):
        vid = cv2.VideoCapture(self.video_source)

        while True:
            # Read frames from the camera
            success, frame = vid.read()

            # Check if frames are successfully captured
            if not success:
                print(f"Error capturing frame from Camera {self.video_source}")
                break

            frame = self.face_dec(frame, self.video_source)

            # Display frames from the camera
            cv2.imshow(f"Camera {self.video_source}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem(video_source=0)
    face_recognition_system1 = FaceRecognitionSystem(video_source=2)
    
    face_recognition_system.run_system()
    face_recognition_system1.run_system()
    
