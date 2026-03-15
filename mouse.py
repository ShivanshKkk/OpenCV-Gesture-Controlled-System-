import cv2
import time
import math
import numpy as np
import pyautogui

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

# ── Safety: stop pyautogui from crashing if mouse hits screen edge
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # no delay between pyautogui calls

# ── Screen & camera config
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480

# ── The region of the camera frame mapped to full screen (inner box = more control)
FRAME_REDUCTION = 100  # px padding on each side

# ── Smoothing: higher = smoother but more lag
SMOOTHING = 7

# ── Distance thresholds (in px within the camera frame)
CLICK_DIST = 35       # index + middle tips close together = left click
RIGHT_CLICK_DIST = 35 # index + thumb tips close together = right click
SCROLL_DIST = 40      # pinky + ring tips close = scroll mode


class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp_drawing
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, mp_hands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return []
        # Thumb (checks x axis)
        fingers.append(1 if self.lmList[4][1] < self.lmList[3][1] else 0)
        # Four fingers (check y axis)
        for tip in [8, 12, 16, 20]:
            fingers.append(1 if self.lmList[tip][2] < self.lmList[tip - 2][2] else 0)
        return fingers

    def distance(self, p1, p2, img=None, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(img, (cx, cy), 8, (255, 255, 0), cv2.FILLED)
        return length, cx, cy


def draw_ui(img, mode, fps, click_cooldown, right_cooldown):
    h, w = img.shape[:2]

    # Draw the active region box
    cv2.rectangle(
        img,
        (FRAME_REDUCTION, FRAME_REDUCTION),
        (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
        (0, 200, 100), 2
    )

    # FPS
    cv2.putText(img, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mode label
    colors = {
        "MOVE":       (0, 255, 150),
        "LEFT CLICK": (0, 100, 255),
        "RIGHT CLICK":(0, 0, 255),
        "SCROLL":     (255, 200, 0),
        "IDLE":       (100, 100, 100),
    }
    color = colors.get(mode, (255, 255, 255))
    cv2.putText(img, mode, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Click flash
    if click_cooldown > 0:
        cv2.circle(img, (w - 30, 30), 15, (0, 100, 255), cv2.FILLED)
    if right_cooldown > 0:
        cv2.circle(img, (w - 60, 30), 15, (0, 0, 255), cv2.FILLED)

    return img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = HandDetector(maxHands=1)

    prev_x, prev_y = 0, 0       # for smoothing
    curr_x, curr_y = 0, 0

    click_cooldown = 0           # frames to show click indicator
    right_cooldown = 0
    scroll_prev_y = None

    pTime = 0
    mode = "IDLE"

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                continue

            img = cv2.flip(img, 1)  # mirror
            h, w = img.shape[:2]

            detector.findHands(img, draw=True)
            lmList = detector.findPosition(img, draw=False)

            if click_cooldown > 0:
                click_cooldown -= 1
            if right_cooldown > 0:
                right_cooldown -= 1

            if lmList:
                fingers = detector.fingersUp()

                # Landmark positions
                ix, iy = lmList[8][1], lmList[8][2]   # index tip
                mx, my = lmList[12][1], lmList[12][2]  # middle tip
                tx, ty = lmList[4][1], lmList[4][2]    # thumb tip

                # ── MODE: INDEX only up = MOVE mouse
                if fingers == [0, 1, 0, 0, 0]:
                    mode = "MOVE"
                    # Map index fingertip from inner box → full screen
                    screen_x = np.interp(
                        ix,
                        [FRAME_REDUCTION, w - FRAME_REDUCTION],
                        [0, SCREEN_W]
                    )
                    screen_y = np.interp(
                        iy,
                        [FRAME_REDUCTION, h - FRAME_REDUCTION],
                        [0, SCREEN_H]
                    )
                    # Smooth
                    curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (screen_y - prev_y) / SMOOTHING
                    prev_x, prev_y = curr_x, curr_y

                    pyautogui.moveTo(curr_x, curr_y)
                    cv2.circle(img, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

                # ── MODE: INDEX + MIDDLE up = LEFT CLICK
                elif fingers == [0, 1, 1, 0, 0]:
                    dist, cx, cy = detector.distance(8, 12, img)
                    if dist < CLICK_DIST and click_cooldown == 0:
                        mode = "LEFT CLICK"
                        pyautogui.click()
                        click_cooldown = 20
                        cv2.circle(img, (cx, cy), 15, (0, 100, 255), cv2.FILLED)
                    else:
                        mode = "MOVE" if dist > CLICK_DIST else "LEFT CLICK"

                # ── MODE: THUMB + INDEX up = RIGHT CLICK
                elif fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0:
                    dist, cx, cy = detector.distance(4, 8, img)
                    if dist < RIGHT_CLICK_DIST and right_cooldown == 0:
                        mode = "RIGHT CLICK"
                        pyautogui.rightClick()
                        right_cooldown = 25
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                    else:
                        mode = "RIGHT CLICK" if dist < RIGHT_CLICK_DIST else "MOVE"

                # ── MODE: ALL fingers up = SCROLL
                elif fingers == [1, 1, 1, 1, 1]:
                    mode = "SCROLL"
                    if scroll_prev_y is not None:
                        delta = scroll_prev_y - iy  # positive = scroll up
                        if abs(delta) > 3:
                            pyautogui.scroll(int(delta / 5))
                    scroll_prev_y = iy

                else:
                    mode = "IDLE"
                    scroll_prev_y = None

            else:
                mode = "IDLE"

            # FPS
            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if (cTime - pTime) > 0 else 0
            pTime = cTime

            img = draw_ui(img, mode, fps, click_cooldown, right_cooldown)

            cv2.imshow("Virtual Mouse", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()