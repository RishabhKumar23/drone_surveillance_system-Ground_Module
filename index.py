import cv2
import socket
import struct
import pickle

# Client configuration
server_ip = '192.168.149.85'
server_port = 5555

# Create client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Open camera
cap = cv2.VideoCapture(0)

# Get Raspberry Pi's IP address
pi_ip = socket.gethostbyname(socket.gethostname())

while True:
    ret, frame = cap.read()

    # Serialize frame
    data = pickle.dumps(frame)

    # Pack frame size and send data
    client_socket.sendall(struct.pack(">L", len(data)) + data)

# Release the camera and close the socket
cap.release()
client_socket.close()