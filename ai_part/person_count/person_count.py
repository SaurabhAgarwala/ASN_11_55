import cv2 as cv
import os, shutil, requests, time

def person_count(img):
    # print(img)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    faces_count = str(len(faces))
    for (x, y, w, h ) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness= 2)
    return faces_count


while(True):
    files = os.listdir('assets')
    for file in files:
        candidate_id = file.split('_')[1]
        cheating = False
        file_path = os.path.join('assets',file)
        image = cv.imread(file_path)
        no_of_persons = person_count(image)
        print("File:", file_path)
        print('No. of persons', no_of_persons)
        # return result
        url = 'http://localhost:8000/proctor/reportviolation/'+candidate_id+'/multiperson/'
        print(url)
        requests.get(url)
        file_destination_path = os.path.join("processed", file)
        os.rename(file_path, file_destination_path)
    time.sleep(15)

