# ==========================================
# Aliasing Artifact Demonstration
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

print("Upload an image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# Original Image
# -------------------------

# Create aliasing by aggressive downsampling
small = cv2.resize(img, (img.shape[1]//8, img.shape[0]//8),
                   interpolation=cv2.INTER_NEAREST)

aliased = cv2.resize(small,
                     (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_NEAREST)

# -------------------------
# Corrective Approach
# Anti-aliasing filter
# -------------------------

blur = cv2.GaussianBlur(img, (7,7), 0)

small_fixed = cv2.resize(blur,
                         (img.shape[1]//8, img.shape[0]//8),
                         interpolation=cv2.INTER_AREA)

corrected = cv2.resize(small_fixed,
                       (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

# -------------------------
# Display
# -------------------------

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(aliased)
plt.title("Aliasing Artifact")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(corrected)
plt.title("Corrected using Anti-Aliasing")
plt.axis("off")

plt.show()

# -------------------------
# Explanation
# -------------------------

print("\nCause of Aliasing:")
print("- Sampling frequency is too low.")
print("- High-frequency image details fold into lower frequencies.")
print("- Produces jagged edges and moire patterns.")

print("\nQuantization Concept:")
print("- Low quantization levels reduce intensity precision.")
print("- This mainly introduces banding/noise rather than aliasing.")

print("\nCorrective Approach:")
print("- Apply Gaussian Blur before downsampling.")
print("- Increase sampling resolution.")
print("- Use proper interpolation (INTER_AREA) while resizing.")
