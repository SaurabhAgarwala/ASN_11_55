import cv2 as cv
import numpy as np
from face_landmarks import get_landmark_model, detect_marks
import dlib

def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also
    find the extreme points
    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype = np.int32)
    mask = cv.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1] + points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_post(end_points, cx, cy):
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    print(x_ratio, y_ratio)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

# def contouring(thresh, mid, img, end_points, right=False):
#     """
#     Find the largest contour on an image divided by a midpoint and subsequently the eye position
#     Parameters
#     ----------
#     thresh : Array of uint8
#         Thresholded image of one side containing the eyeball
#     mid : int
#         The mid point between the eyes
#     img : Array of uint8
#         Original Image
#     end_points : list
#         List containing the exteme points of eye
#     right : boolean, optional
#         Whether calculating for right eye or left eye. The default is False.
#     Returns
#     -------
#     pos: int
#         the position where eyeball is:
#             0 for normal
#             1 for left
#             2 for right
#             3 for up
#     """
#     cnts, _ = cv.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     try:
#         cnt = max(cnts, key = cv.contourArea)
#         M = cv.moments(cnt)
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#         if right:
#             cx += mid
#         cv.circle(img, (cx, cy), 4, (0, 0, 255), 2)
#         pos = find_eyeball_position(end_points, cx, cy)
#         return pos
#     except:
#         pass

def contouring(thresh, mid, img, end_points, right=False):
    cnts,_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv.contourArea)
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print(cx, cy)
        if right:
            cx += mid
        cv.circle(img, (cx, cy), 4, (0, 255, 0), 2)
        pos = find_eyeball_post(end_points, cx, cy)
        print(pos)
        return pos
    except:
        pass

def process_thresh(thresh):
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=4)
    thresh = cv.medianBlur(thresh, 3)
    thresh = cv.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    if left == right and left !=0:
        text= ''
        if left == 1:
            print('Looking left')
            text = 'Looking Left'
            #TODO : Add something
        elif left == 2:
            print('Looking right')
            text = 'Looking Right'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        cv.putText(img, text, (20, 20), cv.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2, cv.LINE_AA
         )
def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    if quantized:
        if modelFile == None:
            modelFile = "model/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "model/opencv_face_detector.pbtxt"
        model = cv.dnn.readNetFromTensorflow(modelFile, configFile)
        
    else:
        if modelFile == None:
            modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "model/deploy.prototxt"
        model = cv.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):

    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

# detector = dlib.get_frontal_face_detector()
face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv.createTrackbar('threshold', 'image', 0, 255, nothing)

while(True):
    ret, img = cap.read()
    # gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 1)
    rects = find_faces(img, face_model)
    for rect in rects:

        # shape = predictor(gray, rect)
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv.dilate(mask, kernel, 5)
        print(end_points_left, end_points_right)

        eyes = cv.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        print(mid)
        eyes_gray = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
        threshold = cv.getTrackbarPos('threshold', 'image')
        _, thresh = cv.threshold(eyes_gray, threshold, 255, cv.THRESH_BINARY)
        thresh = process_thresh(thresh)

        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img,end_points_right, True)
        print(eyeball_pos_left, eyeball_pos_right)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
    cv.imshow('eyes', img)
    cv.imshow("image", thresh)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()