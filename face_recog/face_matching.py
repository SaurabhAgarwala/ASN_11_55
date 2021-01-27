import face_recognition
import dlib
import cv2 as cv
import os
# import numpy as np
# from PIL import Image
# First find the face 

# path = 'ImagesParticipant'
# images = []
# className = []
# myList = os.listdir(path)

# for img in myList :
#     curImg = cv.imread(f'{path}/{img}')
#     images.append(curImg)
#     className.append(os.path.splitext(img)[0])

# print(className)


def load_known_faces(BASE_DIR):  
    known_faces = []
    for filename in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, filename)
        known_faces.append((filename.replace('.jpg',''), path))

    return known_faces



def compare_faces(img1, img2):
    # Load the image
    image1 = face_recognition.load_image_file(img1)
    image2 = face_recognition.load_image_file(img2)

    # Get face encoding
    try:
        image1_encode = face_recognition.face_encodings(image1)[0]
        image2_encode = face_recognition.face_encodings(image2)[0]
        # Compare faces and return True / False
        results = face_recognition.compare_faces([image1_encode], image2_encode)
        print('Encode done....')
        # Return true or false
        return results[0]

    except IndexError as e :
        print(e)
    

    
   
    


def face_rec(file, known_faces_dir):
    known_faces = load_known_faces(known_faces_dir)
    for name, known_file in known_faces:
        if compare_faces(known_file, file):
            # Can do something
            return name
    return 'Unknown'


def main():
    BASE_DIR = 'ImagesParticipant'
    name = face_rec('test.jpg', BASE_DIR)
    print('Test foto : ',name)

if __name__ == "__main__":
    main()



# def findEncodings(images):
    #     encodeList = []
#     for img in images:
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
























"""
The Piece of code below for refference
"""
# encodeListKnownFace = findEncodings(images)
# print('encoding complete')


# # For enable the webcam
# cap = cv.VideoCapture(0)

# while True :
#     success, frame = cap.read()
#     imgS = cv.resize(frame, (0,0), None, 0.25, 0.25)
#     imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnownFace, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnownFace, encodeFace)
#         print(faceDis)
#         print(matches)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = className[matchIndex].upper()
#             print(name)
#             y1, x2, y2, x1 = faceLoc 
#             y1, x2, y2, x1 =  y1 *4, x2 *4, y2*4, x1*4
#             cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv.rectangle(frame, (x1,y2 -35), (x2, y2 -35), (0, 0, 255), cv.FILLED)
#             cv.putText(frame, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#             # You can do something about this

#     cv.imshow('Webcam', frame)
#     cv.waitKey(1)