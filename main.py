import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Get the screen size
screen_width, screen_height = pyautogui.size()

# State variables
dragging = False

def calculate_distance(point1, point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect hands
    results = hands.process(image)

    # Draw hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            h, w, c = image.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Map the coordinates to the screen size
            screen_x = int(screen_width * (x / w))
            screen_y = int(screen_height * (y / h))

            # Move the cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate distances between fingers for gesture recognition
            thumb_index_dist = calculate_distance(thumb_tip, index_finger_tip)
            index_middle_dist = calculate_distance(index_finger_tip, middle_finger_tip)

            # Define gesture thresholds
            click_threshold = 0.05
            drag_threshold = 0.07

            # Click gesture: thumb and index finger close together
            if thumb_index_dist < click_threshold:
                pyautogui.click()

            # Right-click gesture: thumb and index finger close together with index finger MCP
            elif thumb_index_dist < click_threshold and calculate_distance(thumb_tip, index_finger_mcp) < click_threshold:
                pyautogui.rightClick()

            # Drag gesture: index and middle finger close together
            elif index_middle_dist < drag_threshold:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    # Display the resulting frame
    cv2.imshow('Hand Gesture Control', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
