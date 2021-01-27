import cv2 as cv
import numpy as np

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