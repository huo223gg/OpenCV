# -*- coding: utf-8 -*-

import cv2
import numpy as np

cascade_src = 'cars.xml'
#video_src = 'video1.avi'
#video_src = 'video2.avi'
video_src = 'challenge.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower_yel = np.array([20,100,100])
    upper_yel = np.array([30,255,255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masky=cv2.inRange(hsv, lower_yel, upper_yel)
    maskw = cv2.inRange(gray, 200,255)
    maskyw = cv2.bitwise_or(maskw, masky)
    maskyw_image = cv2.bitwise_and(gray, maskyw)

    cv2.imshow('Image', img)
    cv2.imshow('Gray', gray)
    cv2.imshow('White',maskw)
    cv2.imshow('YelORWhi', maskyw)
    cv2.imshow('YelANDWhi', maskyw_image)
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
