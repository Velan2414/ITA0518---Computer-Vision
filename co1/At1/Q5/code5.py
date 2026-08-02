import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload an image
print("Upload a surveillance image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display result
plt.figure(figsize=(8,6))
plt.imshow(img_rgb)
plt.title("Real-Time Surveillance - Face Detection")
plt.axis("off")
plt.show()

# Print information
print("Computer Vision Level : High-Level Vision")
print("Reason :")
print("- Detects human faces.")
print("- Understands objects in the scene.")
print("- Used for surveillance and security.")
print("- Supports real-time monitoring and tracking.")
print(f"\nNumber of faces detected: {len(faces)}")
