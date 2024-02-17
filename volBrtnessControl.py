import cv2
import math
import threading
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get default audio output device and set up volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_volume = vol_range[0]
max_volume = vol_range[1]

# Define brightness control range
min_brightness = 0
max_brightness = 150

def set_volume(dist):
    vol = np.interp(dist, [50, 150], [min_volume, max_volume])
    volume.SetMasterVolumeLevel(vol, None)

def set_brightness(dist):
    brightness = np.interp(dist, [50, 300], [min_brightness, max_brightness])
    sbc.set_brightness(int(brightness))

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Determine handedness based on thumb and pinky finger positions
                thumb_x = hand_landmarks.landmark[4].x
                pinky_x = hand_landmarks.landmark[20].x
                handedness = "Left" if thumb_x > pinky_x else "Right"

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate distance between thumb tip and index finger tip
                x_thumb, y_thumb = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])
                x_index, y_index = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                dist = int(math.sqrt((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2))

                # Adjust volume and brightness based on hand gesture
                if handedness == "Left":
                    t = threading.Thread(target=set_brightness, args=(dist,))
                else:
                    t = threading.Thread(target=set_volume, args=(dist,))
                t.start()

        # Display the frame
        cv2.imshow('Hand Gesture Control', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
