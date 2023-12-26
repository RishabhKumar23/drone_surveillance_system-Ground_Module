# drone_surveillance_system-Ground_Module
# Drone Surveillance System => Ground Module

This Python script implements a face recognition attendance system using the `face_recognition` library along with other dependencies such as OpenCV (`cv2`), `numpy`, `os`, `datetime`, `gtts` (Google Text-to-Speech), `pyttsx3`, and `zmq` (ZeroMQ). The system captures frames from two cameras, detects faces, recognizes them based on preloaded images, and marks attendance if a recognized face is authorized.

## Prerequisites
- Python 3
- Install the required libraries using:
  ```
  pip install numpy opencv-python face-recognition gtts pyttsx3 zmq
  ```

## Usage
1. Organize your authorized persons' images in the `authorization_persons` directory.
2. Run the script.

## Features
- **Face Detection and Recognition**: Utilizes the `face_recognition` library to detect and recognize faces from the input frames.
- **Attendance Marking**: Marks the attendance of authorized persons by writing their names and the current time to a CSV file (`Timing.csv`).
- **Text-to-Speech**: Welcomes authorized persons using text-to-speech synthesis with Google Text-to-Speech (`gtts`).
- **Multi-Camera Support**: The script captures frames from two cameras simultaneously.

## File Descriptions
- `FaceRecognitionAttendance.py`: The main Python script containing the face recognition attendance system.
- `Timing.csv`: CSV file storing attendance data with columns "Name" and "Time".

## Dependencies
- `cv2`: OpenCV library for image and video processing.
- `numpy`: Numerical operations library for Python.
- `face_recognition`: Face recognition library built on top of `dlib`.
- `gtts`: Google Text-to-Speech library for text-to-speech synthesis.
- `pyttsx3`: Python Text-to-Speech library for offline text-to-speech synthesis.
- `zmq`: ZeroMQ library for messaging.

## How to Contribute
Feel free to contribute by opening issues, suggesting improvements, or submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
- [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey.
- OpenCV (https://opencv.org/)

**Note**: Make sure to install the required dependencies before running the script. You may need to install additional libraries based on your system configuration.
