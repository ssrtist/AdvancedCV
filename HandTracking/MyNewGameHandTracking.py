import mediapipe as mp
import time
import cv2
import HandTrackingModule as ht

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = ht.handDetector()
while True:
    success, img = cap.read()
    cv2.flip(img, 1, img)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0: 
        print(lmList[4])

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):break
