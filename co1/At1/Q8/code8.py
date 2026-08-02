import cv2
import numpy as np
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
# Add Artificial Gaussian Noise
# ---------------------------------------

noise = np.random.normal(0, 25, img.shape).astype(np.int16)

noisy = img.astype(np.int16) + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# ---------------------------------------
# Remove Noise using Gaussian Blur
# ---------------------------------------

denoised = cv2.GaussianBlur(noisy, (5,5), 0)

# ---------------------------------------
# Display Images
# ---------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(denoised)
plt.title("Denoised Image")
plt.axis("off")

plt.show()

# ---------------------------------------
# Explanation
# ---------------------------------------

print("Image Acquisition Noise Sources:")
print("- Low light")
print("- Sensor electronic noise")
print("- High ISO")
print("- Thermal noise")
print("- Transmission errors")

print("\nNoise Reduction Techniques:")
print("- Gaussian Filter")
print("- Median Filter")
print("- Better camera sensors")
print("- Improved lighting")
print("- Lower ISO settings")
