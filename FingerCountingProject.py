import cv2
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.75)

# Finger tip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        fingers = []

        # Thumb: Check if tip is to the right of its lower joint
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Remaining 4 fingers: Check if tip is above the middle joint
        for i in range(1, 5):
            if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        cv2.putText(img, f'Fingers: {totalFingers}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Finger Counting", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
