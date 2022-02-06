import numpy as np
import cv2
import dlib
import pyautogui as pag
import imutils
from imutils import face_utils

def _EAR(eye):
    e1 = np.linalg.norm(eye[1] - eye[5])
    e2 = np.linalg.norm(eye[2] - eye[4])
    e3 = np.linalg.norm(eye[0] - eye[3])
    ear = (e1 + e2) / (2.0 * e3)
    return ear

def _MAR(mouth):
    m1 = np.linalg.norm(mouth[13] - mouth[19])
    m2 = np.linalg.norm(mouth[14] - mouth[18])
    m3 = np.linalg.norm(mouth[15] - mouth[17])
    m4 = np.linalg.norm(mouth[12] - mouth[16])
    mar = (m1 + m2 + m3) / (3 * m4)
    return mar

def _DetermineDirection(nose_point, nosePoint, w, h, multiple=1):
    nx, ny = nose_point
    x, y = nosePoint
    R_L = ""
    U_D = ""
    if nx > x + multiple * w:
        R_L ="R"
    elif nx < x - multiple * w:
        R_L ="L"

    if ny > y + multiple * h:
        U_D="D"
    elif ny < y - multiple * h:
        U_D="U"
    return R_L,U_D

eyeCounter = 0
mouthCounter = 0
winkCount = 0
scrollMode = False
inputMode = False
nosePoint = (0,0)

detector = dlib.get_frontal_face_detector()
predictor = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor)

vid = cv2.VideoCapture(0)
cam_w = cam_h  = 700
unit_w = 2
unit_h = 1

while True:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    mouth = shape[48:68]
    rightEye = shape[42:48]
    leftEye = shape[36:42]
    nose = shape[27:36]
    nose_point = (nose[3, 0], nose[3, 1])

    MAR = _MAR(mouth)
    LEAR = _EAR(leftEye)
    REAR = _EAR(rightEye)
    EAR = (LEAR + REAR) / 2.0
    DIFF_EAR = np.abs(LEAR - REAR)

    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 255), 1)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)

    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
    if DIFF_EAR > 0.04:
        if LEAR < REAR:
            if LEAR < 0.20:
                winkCount += 1
                if winkCount > 5:
                    pag.click(button='left')
                    winkCount = 0
        elif LEAR > REAR:
            if REAR < 0.20:
                winkCount += 1
                if winkCount > 5:
                    pag.click(button='right')
                    winkCount = 0
        else:
            winkCount = 0
    else:
        if EAR <= 0.20:
            eyeCounter += 1
            if eyeCounter > 15:
                scrollMode = not scrollMode
                eyeCounter = 0
        else:
            eyeCounter = 0
            winkCount = 0
    if MAR > 0.5:
        mouthCounter += 1
        if mouthCounter >= 15:
            inputMode = not inputMode
            mouthCounter = 0
            nosePoint = nose_point
    else:
        mouthCounter = 0
    if inputMode:
        cv2.putText(frame, "INPUT ENABLED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        x, y = nosePoint
        nx, ny = nose_point
        w, h = 35, 20
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.line(frame, nosePoint, nose_point, (255, 0, 0), 2)
        R_L,U_D = _DetermineDirection(nose_point, nosePoint, w, h)
        cv2.putText(frame, R_L + U_D, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        drag = 25
        if U_D == 'U':
            if scrollMode:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif U_D == 'D':
            if scrollMode:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)
        if R_L == 'L':
            pag.moveRel(-drag, 0)
        elif R_L == 'R':
            pag.moveRel(drag, 0)

    if scrollMode:
        cv2.putText(frame, 'SCROLL MODE ENABLED', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cv2.destroyAllWindows()
vid.release()
