# current progress: (27:00) Hand Tracking - Module
# https://youtu.be/01sAkU_NvOY?list=FLmQacUzewqgkxFdg9JOtIjw&t=1619

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    cv2.flip(img, 1, img)

    imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==0:
                    print(id,': ',cx,cy)
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)

            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #print(handLms)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):break

