import cv2
import os
import numpy as np

# Create a directory to store face images
dataset_dir = 'face_dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Step 1: Capture images of the face and save them with the label "Joshua"
def capture_face_images(name='Joshua', num_images=50):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_name = f"{dataset_dir}/{name}_{count}.jpg"
            cv2.imwrite(file_name, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {name} - {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images of {name}")


# Step 2: Train the face recognizer model
def train_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []

    # Load images and labels from dataset
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_dir, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(gray_img)
            labels.append(0)  # All images are labeled with ID 0 (representing "Joshua")

    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognizer.yml')
    print("Model trained and saved as 'face_recognizer.yml'")


# Step 3: Detect and recognize the face using the trained model
def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognizer.yml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)

            if label == 0 and confidence < 50:  # Lower confidence means better match
                name = "Joshua"
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                name = "Unknown"
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the steps
print("Starting face capture...")
capture_face_images()

print("Training the model...")
train_face_recognizer()

print("Starting face recognition...")
recognize_face()

