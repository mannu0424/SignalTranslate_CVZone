import math
import time
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
handDetector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = 'DataSet/F'
count = 0

while cap.isOpened():
    success, img = cap.read()
    hands, img = handDetector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCalc = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeCrop = imgResize.shape
            wGap = math.ceil((imgSize - wCalc) / 2)
            imgWhite[:, wGap:wCalc + wGap] = imgResize

        else:
            k = imgSize / w
            hCalc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalc))
            imgResizeCrop = imgResize.shape
            hGap = math.ceil((imgSize - hCalc) / 2)
            imgWhite[hGap:hCalc + hGap, :] = imgResize

        cv2.imshow('imgCrop', imgCrop)
        cv2.imshow('imgWhite', imgWhite)

    cv2.imshow('img', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

    elif key == ord('s'):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)

cap.release()
cv2.destroyAllWindows()
