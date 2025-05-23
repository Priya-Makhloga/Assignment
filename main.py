
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_frame

model = load_model('saved_model/model.h5')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)
    prediction = np.argmax(model.predict(processed), axis=-1)[0]

    cv2.putText(frame, f'Prediction: {prediction}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
