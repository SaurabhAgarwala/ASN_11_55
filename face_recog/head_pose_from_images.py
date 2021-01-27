import os
import cv2 as cv
import sys
import argparse
import numpy as np
import dlib

from draw_face import draw

import reference_world as world

"""
Kalo misalkan posisi kepalanya
ngga straight anggep terindikasi
nyontek jika lebih dari 5 detik.

"""

PREDICTOR_PATH = os.path.join('models','shape_predictor_68_face_landmarks.dat')

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] Please download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", type=str, help= "image location for pose estimation")

parser.add_argument("-f", "--focal", type=float, help= "Callibrated Focal Length of the Camera")

args = parser.parse_args()

def main(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    isCheat = True
    while isCheat :
        img = cv.imread(image)
        faces = detector(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

        face3Dmodel = world.ref3DModel()
        
        for face in faces :
            newImg= cv.cvtColor(img, cv.COLOR_BGR2RGB)
            shape = predictor(newImg, face)

            draw(img, shape)

            refImgPts = world.ref2dImagePoints(shape)

            h, w, c = img.shape

            focalLength = args.focal * w
            
            camMatrix = world.cameraMatrix(focalLength, (h/2, w/2))

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

            print('*', 80)
            print('Angle: ', angles)

            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])

            print("Axis X : ", x)
            print("Axis Y :", y)
            print("Axis Z :", z)
            print('*' * 80)

            gaze = "Looking : "
            if angles[1] < -15 :
                gaze += "Left"
                print('Please Look forward, do not cheating')
            elif angles[1] > 15:
                gaze+= "Right"
                print('Please Look Forward, do not cheating')
            else :
                gaze += "Forward"
                isCheat = False
                
            cv.putText(img, gaze, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv.imshow("Head Pose", img)

            
            key = cv.waitKey(0) & 0xFF

        if key == 27:
            cv.imwrite(f"joye-{gaze}.jpg", img)
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(args.image)