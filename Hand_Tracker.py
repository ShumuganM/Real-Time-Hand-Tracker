import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the downloaded hand landmarker model
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

# Map fingertip landmark IDs to names and BGR colors
FINGERS = {
    4:  ("Thumb",  (255, 0, 255)),   # Magenta
    8:  ("Index",  (0, 255, 0)),     # Green
    12: ("Middle", (255, 255, 0)),   # Cyan
    16: ("Ring",   (0, 165, 255)),   # Orange
    20: ("Pinky",  (255, 100, 100))  # Light Blue
}

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

print("Starting camera... Press 'q' to quit.")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        # Flip horizontally for a mirror view and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp_ms += 33
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if results.hand_landmarks:
            h, w, _ = frame.shape

            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                # Identify Left vs. Right hand (flipped because of mirror view)
                raw_label = results.handedness[hand_idx][0].category_name
                hand_label = "Left" if raw_label == "Right" else "Right"

                # Draw skeleton lines
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)

                # Draw joints and label all 5 fingertips
                for idx, (cx, cy) in enumerate(points):
                    if idx in FINGERS:
                        name, color = FINGERS[idx]
                        cv2.circle(frame, (cx, cy), 10, color, cv2.FILLED)
                        cv2.putText(
                            frame, name, (cx + 12, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )
                    else:
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

                # Detect which fingers are raised
                raised_fingers = []

                # Thumb check (horizontal x-position relative to thumb joint)
                if hand_label == "Right":
                    if points[4][0] < points[3][0]:
                        raised_fingers.append("Thumb")
                else:
                    if points[4][0] > points[3][0]:
                        raised_fingers.append("Thumb")

                # Index, Middle, Ring, Pinky check (tip y-position above middle joint)
                for tip_id in [8, 12, 16, 20]:
                    if points[tip_id][1] < points[tip_id - 2][1]:
                        raised_fingers.append(FINGERS[tip_id][0])

                # Display active fingers at the top of the screen
                status_text = f"{hand_label} Hand: {', '.join(raised_fingers) if raised_fingers else 'Closed Fist'}"
                y_offset = 40 + (hand_idx * 35)
                cv2.putText(
                    frame, status_text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

        cv2.imshow("All Fingers Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()