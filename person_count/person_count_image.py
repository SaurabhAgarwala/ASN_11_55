import cv2
face_cascade = cv2.CascadeClassifier("dataset\haarcascade_frontalface_default.xml")
img = cv2.imread("image_people/image.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
faces_count = str(len(faces))

for (x, y, w, h ) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness= 2)
print(faces_count + " people")
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)