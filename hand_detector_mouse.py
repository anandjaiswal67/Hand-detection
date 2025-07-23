import cv2
import mediapipe as mp
import pyautogui
import math

wCam, hCam = 340, 280
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

clicked = False
prev_x, prev_y = 0, 0
prev_scroll_y = None
smooth_factor = 0.2
dpi_scaling = 1.5
scroll_threshold = 15  # pixels

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lmList.append((cx, cy))

            if lmList:
                # Cursor movement
                x, y = lmList[8]
                screen_x = int(screen_w * x / wCam)
                screen_y = int(screen_h * y / hCam)
                new_x = prev_x + (screen_x - prev_x) * smooth_factor * dpi_scaling
                new_y = prev_y + (screen_y - prev_y) * smooth_factor * dpi_scaling
                pyautogui.moveTo(new_x, new_y)
                prev_x, prev_y = new_x, new_y

                # Scroll gesture
                curr_scroll_y = lmList[8][1]
                if prev_scroll_y is not None:
                    delta_y = curr_scroll_y - prev_scroll_y
                    if abs(delta_y) > scroll_threshold:
                        scroll_amount = -1 if delta_y < 0 else 1
                        pyautogui.scroll(scroll_amount * 30)  # Adjust scroll speed
                prev_scroll_y = curr_scroll_y

                # Left-click gesture
                x1, y1 = lmList[8]
                x2, y2 = lmList[12]
                distance = math.hypot(x2 - x1, y2 - y1)

                if distance < 30 and not clicked:
                    pyautogui.click(button='left')
                    clicked = True
                elif distance >= 30:
                    clicked = False

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Mouse with Scroll", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()