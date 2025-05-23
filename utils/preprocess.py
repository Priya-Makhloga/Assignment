
import cv2
import numpy as np

def preprocess_frame(frame, target_size=(28, 28)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size).reshape(1, target_size[0], target_size[1], 1) / 255.0
    return resized
