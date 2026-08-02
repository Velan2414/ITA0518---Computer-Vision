import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
print("Upload an image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------
# Low Sampling (Reduce Resolution)
# ---------------------------------------

low_res = cv2.resize(
    img,
    (img.shape[1]//8, img.shape[0]//8),
    interpolation=cv2.INTER_AREA
)

low_res = cv2.resize(
    low_res,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# ---------------------------------------
# Improve Image Formation
# Sharpen Image
# ---------------------------------------

kernel = [[0,-1,0],
          [-1,5,-1],
          [0,-1,0]]

kernel = cv2.UMat.get(cv2.UMat(kernel)) if False else __import__("numpy").array(kernel)

sharp = cv2.filter2D(img, -1, kernel)

# ---------------------------------------
# Display
# ---------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(low_res)
plt.title("Low Resolution Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(sharp)
plt.title("Enhanced Image")
plt.axis("off")

plt.show()

# ---------------------------------------
# Explanation
# ---------------------------------------

print("Sampling:")
print("- Low sampling reduces image resolution.")
print("- Objects become blurry and difficult to detect.")

print("\nImage Formation:")
print("- Camera quality, focus, and lighting affect image clarity.")
print("- Sharpening improves edge visibility.")

print("\nApplications:")
print("- Autonomous driving")
print("- Surveillance")
print("- Medical imaging")
