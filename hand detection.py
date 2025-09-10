import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Tips of each finger
finger_tips_ids = [4, 8, 12, 16, 20]

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror view
    h, w, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_x, lm_y = int(lm.x * w), int(lm.y * h)
                lm_list.append((lm_x, lm_y))

            if lm_list:
                fingers = []

                # Thumb (check x direction due to orientation)
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers
                for tip_id in finger_tips_ids[1:]:
                    if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)

                # Gesture Logic
                # Thumbs Up
                if fingers == [1, 0, 0, 0, 0] and lm_list[4][1] < lm_list[3][1]:
                    cv2.putText(frame, "👍 Thumbs Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

                # Thumbs Down
                elif fingers == [1, 0, 0, 0, 0] and lm_list[4][1] > lm_list[3][1]:
                    cv2.putText(frame, "👎 Thumbs Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

                # Finger Counting
                else:
                    cv2.putText(frame, f"Fingers: {total_fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)

    # Show result
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
