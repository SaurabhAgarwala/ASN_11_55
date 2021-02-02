import cv2 as cv
import numpy as np
import dlib
import shutil
import os
import sys
from glob import glob
import face_recognition
# import object_detect as ob

def drawPolyline(img, shapes, start, end, isClosed= False):
    points= []
    for i in range(start, end+1):
        point =[shapes.part(i).x, shapes.part(i).y]
        points.append(point)
    points = np.array(points, dtype = np.int32)
    cv.polylines(img, [points], isClosed, (125, 200, 0),
    thickness=1, lineType=cv.LINE_8)


def draw(img, shapes):
    drawPolyline(img, shapes, 0, 16)
    drawPolyline(img, shapes, 17, 21)
    drawPolyline(img, shapes, 22, 26)
    drawPolyline(img, shapes, 27, 30)
    drawPolyline(img, shapes, 30, 35, True)
    drawPolyline(img, shapes, 36, 41, True)
    drawPolyline(img, shapes, 42, 47, True)
    drawPolyline(img, shapes, 48, 59, True)
    drawPolyline(img, shapes, 60, 67, True)


def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)

def ref2dImagePoints(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)


def cameraMatrix(fl, center):
    mat = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(mat, dtype=np.float)


def gaze_tracking(img, PREDICTOR_PATH, focal=1):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    faces = detector(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

    face3Dmodel = ref3DModel()
    count = 0
        
    for face in faces :
        newImg= cv.cvtColor(img, cv.COLOR_BGR2RGB)
        shape = predictor(newImg, face)

        draw(img, shape)

        refImgPts = ref2dImagePoints(shape)

        h, w, c = img.shape

        focalLength = focal * w
            
        camMatrix = cameraMatrix(focalLength, (h/2, w/2))

        mdist = np.zeros((4,1), dtype= np.float64)

         # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv.solvePnP(face3Dmodel, refImgPts, camMatrix, mdist)

        noseEndPoints3D= np.array([[0, 0, 1000.0]], dtype= np.float64)
        noseEndPoints2D, jacobian = cv.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, camMatrix, mdist
            )

        # Draw nose line

        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoints2D[0, 0, 0]), int(noseEndPoints2D[0,0, 1]))

        cv.line(img, p1, p2, (110, 220, 0), thickness=2, lineType= cv.LINE_AA)

        ## Calculatiing angle
        rmat, jac = cv.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

        # print('*' * 80)
        # print('Angle: ', angles)

        # x = np.arctan2(Qx[2][1], Qx[2][2])
        # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        # z = np.arctan2(Qz[0][0], Qz[1][0])

        # print("Axis X : ", x)
        # print("Axis Y :", y)
        # print("Axis Z :", z)
        # print('*' * 80)

        gaze = "Looking "
        if angles[1] < -10 :
            gaze += "Left"
            cv.putText(img, gaze, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv.imshow("Head Pose", img)
            cv.waitKey(20)
            print('Please Look forward, do not cheating')
            return True, 'Looking : Left'
        
            
        elif angles[1] > 10:
            gaze+= "Right"
            
            cv.putText(img, gaze, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv.imshow("Head Pose", img)
            cv.waitKey(20)
            print('Please Look Forward, do not cheating')
            return True, 'Looking : Right'

        else :
            gaze += "Forward"
            cv.putText(img, gaze, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv.imshow("Head Pose", img)
            cv.waitKey(20)
            return False, 'Looking : Forward'
        
        cv.putText(img, gaze, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv.imshow("Head Pose", img)

            

def person_count(img):
    face_cascade = cv.CascadeClassifier("model\haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    faces_count = str(len(faces))
    for (x, y, w, h ) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness= 2)
    return img, faces_count

def load_known_faces(BASE_DIR='known_faces'):  
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
    

list_images = glob('stream_snapshot/*.jpg')
PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
destination_path = 'processed_image'
KNOWN_FACES_DIR = "known_faces"
results = []
indicates_violation=[]
uknown_images=[]

known_faces = load_known_faces()
known_faces = glob('known_faces/*.jpg')
for img in list_images:
    image = cv.imread(img)
    image, faces_count = person_count(image)
    img_split = img.split('\\')
    img_name = img_split[1]
    participant_id = img_name.split('.')
    gaze = ''
    prep_img = ob.preprocess_img(image)
    result_img = ob.predict(image, prep_img)
    cv.imshow('Object Detect', result_img)
    cv.waitKey(100)
    for known_img in known_faces:
        if compare_faces(known_img, img):
            prep_img = ob.preprocess_img(image)
            result_img = ob.predict(image, prep_img)
            cv.imshow('Object Detect', result_img)
            cv.waitKey(50)
            if int(faces_count) > 1:
                print("Multiple Person Detected!")
                indicates_violation.append(participant_id[0])
                # gaze='Cannot detect directions of multiple faces'
                continue
            else :
                is_cheat, gaze = gaze_tracking(image, PREDICTOR_PATH)
                if is_cheat:
                    indicates_violation.append(participant_id[0]) 
            print(f'Image : {img_name}, faces count :{faces_count}, {gaze}')     
        else:
            uknown_images.append(img_name)
            continue
    
    file_destination_path = os.path.join(destination_path, img_name)
    os.rename(img, file_destination_path)
    shutil.move(file_destination_path, img)
    os.replace(img, file_destination_path)
print('Candidates ID that indicates cheat: ',indicates_violation)
print('Unknown Images :', uknown_images)
# print(compare_faces('known_faces/haikal.jpg', 'stream_snapshot/2345.jpg'))
# gaze_tracking(cv.imread('stream_snapshot/2345.jpg'), PREDICTOR_PATH)
cv.waitKey(0)
sys.exit()