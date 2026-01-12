import cv2                  # OpenCV → used to open webcam and show video
import mediapipe as mp      # MediaPipe → used to detect hand and fingers
import pyautogui            # PyAutoGUI → used to control mouse

# -------------------- BASIC SETUP --------------------

# Open laptop webcam (0 means default camera)
cap = cv2.VideoCapture(0)

# Load MediaPipe hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)   # we track only ONE hand
mp_draw = mp.solutions.drawing_utils      # used to draw dots and lines

# Get laptop screen size (mouse moves on screen, not camera)
screen_w, screen_h = pyautogui.size()

# Variables to make mouse movement smooth (reduce shaking)
prev_x, prev_y = 0, 0
smoothening = 5

# This variable remembers whether mouse is already clicked or not
is_clicking = False

# -------------------- IMPORTANT DISTANCE VALUES --------------------
# Smaller distance → fingers are pinched
# Larger distance → fingers are open

CLICK_DISTANCE = 0.025     # fingers must be VERY close to start holding
RELEASE_DISTANCE = 0.045   # fingers must be clearly separated to release

# Safety feature: move mouse to top-left corner to stop program
pyautogui.FAILSAFE = True

# -------------------- MAIN LOOP (RUNS CONTINUOUSLY) --------------------

while True:
    success, frame = cap.read()     # Read one frame from webcam
    if not success:
        break

    # Flip the camera image so movement feels natural (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert image from BGR to RGB (MediaPipe only understands RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ask MediaPipe to detect hand and finger points
    results = hands.process(frame_rgb)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Get index finger tip (point number 8)
            index_tip = hand_landmarks.landmark[8]

            # Get thumb finger tip (point number 4)
            thumb_tip = hand_landmarks.landmark[4]

            # Convert index finger position from percentage to screen pixels
            mouse_x = int(index_tip.x * screen_w)
            mouse_y = int(index_tip.y * screen_h)

            # Smooth mouse movement (avoid sudden jumps)
            curr_x = prev_x + (mouse_x - prev_x) / smoothening
            curr_y = prev_y + (mouse_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)

            # Save current position for next frame
            prev_x, prev_y = curr_x, curr_y

            # Calculate distance between thumb and index finger
            # This tells us whether fingers are pinched or open
            distance = ((thumb_tip.x - index_tip.x) ** 2 +
                        (thumb_tip.y - index_tip.y) ** 2) ** 0.5

            # ---------------- CLICK AND HOLD LOGIC ----------------

            # If fingers are pinched AND mouse is not already clicked
            if not is_clicking and distance < CLICK_DISTANCE:
                pyautogui.mouseDown()     # Hold mouse click
                is_clicking = True        # Remember that mouse is held

            # If fingers are opened clearly AND mouse is currently held
            elif is_clicking and distance > RELEASE_DISTANCE:
                pyautogui.mouseUp()       # Release mouse click
                is_clicking = False       # Update click state

            # Draw hand dots and finger connections on camera screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Show current action on screen (for understanding)
            status = "HOLD" if is_clicking else "MOVE"
            cv2.putText(frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    # Show webcam window
    cv2.imshow("Gesture Controlled Mouse", frame)

    # Press 'q' key to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEANUP --------------------

# Turn off webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
