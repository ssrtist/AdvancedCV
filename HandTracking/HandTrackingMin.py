import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #print(handLms)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):break

