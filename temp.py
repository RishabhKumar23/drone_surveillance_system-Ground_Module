import cv2
import socket
import pickle
import struct
from datetime import datetime
import os
import numpy as np
from gtts import gTTS
import face_recognition as face_rec

# Server configuration
server_ip = 'localhost'  # Change to the actual IP address of the server
server_port = 6000

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)

print(f"Server listening on {server_ip}:{server_port}")

# Accept a connection from the client
client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")

# Load student images and encodings
path = 'authorization_persons'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

EncodeList = []
for img in studentImg:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodeimg = face_rec.face_encodings(img)[0]
    EncodeList.append(encodeimg)

# Start capturing video
vid = cv2.VideoCapture(0)

while True:
    try:
        # Capture a frame
        success, frame = vid.read()

        # Resize frame for faster processing
        smaller_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

        # Encode faces in the frame
        faces_in_frame = face_rec.face_locations(smaller_frame)
        encode_faces_in_frame = face_rec.face_encodings(smaller_frame, faces_in_frame)

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
                # Mark attendance and generate welcome message
                with open('Timing.csv', 'a') as f:
                    now = datetime.now()
                    timestr = now.strftime('%H:%M')
                    f.write(f'\n{name}, {timestr}')
                    statement = f'Welcome to class, {name}'
                    tts = gTTS(text=statement, lang='en')
                    tts.save('welcome.mp3')
                    os.system('mpg321 welcome.mp3')  # install mpg321

        # Display the processed frame
        cv2.imshow("Processed Frame", frame)

        # Serialize the frame
        frame_data = pickle.dumps(frame)
        msg_size = struct.pack("L", len(frame_data))

        # Send the frame size
        client_socket.sendall(msg_size)

        # Send the serialized frame
        client_socket.sendall(frame_data)

    except socket.error as e:
        print(f"Socket error: {e}")
        break

    except pickle.PickleError as e:
        print(f"Error pickling frame: {e}")
        continue  # Skip this iteration and continue with the next loop

    except Exception as e:
        print(f"Error: {e}")
        break

# Release resources
client_socket.close()
server_socket.close()
cv2.destroyAllWindows()
