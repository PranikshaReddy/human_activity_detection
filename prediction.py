import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r'webapp\models\MobileNet.h5',compile=False)

# Set up video capture device
cap = cv2.VideoCapture(0)

# Define labels
labels = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting', 'hugging',
          'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop']


def livestreaming():
    while True:
        # Read frame from video capture device
        ret, frame = cap.read()

        # Preprocess the image
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction
        prediction = model.predict(input_frame)[0]

        # Get predicted label
        predicted_label = labels[np.argmax(prediction)]

        # Overlay label onto image
        cv2.putText(frame, predicted_label, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show video frame
        cv2.imshow('frame', frame)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture device and close window
    cap.release()
    cv2.destroyAllWindows()
