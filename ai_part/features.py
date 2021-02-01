import cv2 as cv
import numpy as np
import dlib
import shutil
import os
import sys
from glob import glob

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
    face_cascade = cv.CascadeClassifier("models\haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    faces_count = str(len(faces))
    for (x, y, w, h ) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), color=(0, 255, 0), thickness= 2)
    return img, faces_count

list_images = glob('stream_snapshot/*.jpg')
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
destination_path = 'processed_image'
results = []
indicates_violation=[]
# img_split = list_images[0].split('\\')
# print(img_split)

for img in list_images:
    image = cv.imread(img)
    image, faces_count = person_count(image)
    img_split = img.split('\\')
    img_name = img_split[1]
    participant_id = img_name.split('.')
    gaze = ''
    if int(faces_count) > 1:
        print("Multiple Person Detected!")
        indicates_violation.append(participant_id[0])
        # gaze='Cannot detect directions of multiple faces'
        # continue
    else :
        is_cheat, gaze = gaze_tracking(image, PREDICTOR_PATH)
        if is_cheat:
            indicates_violation.append(participant_id[0])
            
    print(f'Image : {img_name}, faces count :{faces_count}, {gaze}')
    file_destination_path = os.path.join(destination_path, img_name)
    os.rename(img, file_destination_path)
    shutil.move(file_destination_path, img)
    os.replace(img, file_destination_path)
print('Candidates ID that indicates cheat: ',indicates_violation)

sys.exit()