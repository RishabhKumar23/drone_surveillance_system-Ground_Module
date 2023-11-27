# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
Img = face_rec.load_image_file('database/Rishabh.jpeg')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
Img = resize(Img, 0.50)
Img_test = face_rec.load_image_file('database/Rishabh_test.jpeg')
Img_test = resize(Img, 0.50)
Img_test = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

# finding face location
faceLocation_img = face_rec.face_locations(Img)[0]
encode_img = face_rec.face_encodings(Img)[0]
cv2.rectangle(Img, (faceLocation_img[3], faceLocation_img[0]), (faceLocation_img[1], faceLocation_img[2]), (255, 0, 255), 3)


faceLocation_Img_test = face_rec.face_locations(Img_test)[0]
encode_Img_test = face_rec.face_encodings(Img_test)[0]
cv2.rectangle(Img_test, (faceLocation_img[3], faceLocation_img[0]), (faceLocation_img[1], faceLocation_img[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_img], encode_Img_test)
print(results)
cv2.putText(Img_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', Img)
cv2.imshow('test_img', Img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()