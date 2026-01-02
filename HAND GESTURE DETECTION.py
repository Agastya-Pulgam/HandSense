import cv2
import time
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker

FIST = "FIST"
PALM = "PALM"
UNKNOWN = "UNKNOWN"
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGER_PAIRS = [
    (8,6),
    (12,10),
    (16,14),
    (20,18)
]
KEY_LANDMARKS = {
    "WRIST": 0,
    "INDEX_TIP": 8,
    "INDEX_PIP": 6,
    "MIDDLE_TIP": 12,
    "MIDDLE_PIP": 10
}
BOX_POINTS = [0,8,12,16,20]

fps = 0

def extract(hand_landmarks, indices):
    return {i: hand_landmarks[i] for i in indices}
def detect_gestures(hand_landmarkers):

    palm_y = hand_landmarkers[WRIST].y
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    fingers_up = sum(hand_landmarkers[t].y < palm_y for t in tips)

    if fingers_up >= 3:
        return PALM
    if fingers_up == 0:
        return FIST

def main():

    cap = cv2.VideoCapture(1)

    if not cap.isOpened(): 
        print('Error: could not open webcam')
        return
    
    mp_hands = mp.tasks.vision.HandLandmarker
    num_hands = 2
    options = vision.HandLandmarkerOptions(base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands = 2
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(options)
    
    last_gesture = None
    gesture_count = 0
    STABLE_FRAMES = 8
    counter = 0
    prev_time = time.time()
    fps = 0.0
    while True:

        curr_time = time.time()
        if curr_time - prev_time > 1e-6:
            inst_fps = curr_time - prev_time
            fps = 0.9 * fps + 0.1 * (1 / inst_fps)
        prev_time = curr_time
        success, frame = cap.read()
        if not success:
            print("failed to read frames")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640,480))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = hand_landmarker.detect_for_video(
            mp_image,
            int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        )
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                lm = extract(hand_landmarks, [0,8,12,16,20])
                gesture = detect_gestures(lm)
                for lm in hand_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    xs = [int(lm.x * w) for lm in hand_landmarks]
                    ys = [int(lm.y * h) for lm in hand_landmarks]

                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0,255,0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Gesture: {detect_gestures(hand_landmarks)}",
                        (x_min,y_min + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"FPS: {int(fps)}",
                        (30,30),
                        cv2.FORMATTER_FMT_MATLAB,
                        1,
                        (0,255,255),
                        2
                    )
        cv2.imshow("GESTURE: ",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for hand_landmarks in result.hand_landmarks:
            gesture = detect_gestures(hand_landmarks)
            if gesture == last_gesture:
                gesture_count += 1
            else:
                last_gesture = gesture
                gesture_count = 1

            # if gesture_count == STABLE_FRAMES:
                # print(f"STABLE GESTURE: {gesture}")
                # if gesture == PALM:
                    # counter += 1
                    # cv2.putText(
                        # frame,
                        # f"Gesture: {gesture}",
                        # (30,50),
                        # cv2.FONT_HERSHEY_SIMPLEX,
                        # 1,
                        # (0,255,0),
                        # 2
                    # )
                    # cv2.imshow("Cam",frame)
                # elif gesture == FIST:
                    # counter -= 1
                    # cv2.putText(
                        # frame,
                        # f"Gesture: {gesture}",
                        # (30,50),
                        # cv2.FONT_HERSHEY_SIMPLEX,
                        # 1,
                        # (0,255,0),
                        # 2
                    # )
                    # cv2.imshow("GESTURE: ",frame)



    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    