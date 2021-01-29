import cv2
face_cascade = cv2.CascadeClassifier("dataset\haarcascade_frontalface_default.xml")
img = cv2.imread("image_people/image.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
faces_count = str(len(faces))
print(faces_count + " people")