import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload an image
print("Upload an image")
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Read image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# Simulate CCD (Low Noise)
# --------------------------------------------------

ccd = cv2.GaussianBlur(img, (3,3), 0)

# --------------------------------------------------
# Simulate CMOS (More Noise)
# --------------------------------------------------

noise = np.random.normal(0, 20, img.shape).astype(np.int16)
cmos = img.astype(np.int16) + noise
cmos = np.clip(cmos, 0, 255).astype(np.uint8)

# --------------------------------------------------
# Display Images
# --------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(ccd)
plt.title("CCD (Low Noise)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cmos)
plt.title("CMOS (Higher Noise)")
plt.axis("off")

plt.show()

# --------------------------------------------------
# Comparison
# --------------------------------------------------

print("CCD Sensor")
print("- High image quality")
print("- Low noise")
print("- Better low-light performance")
print("- Higher cost and power consumption")

print("\nCMOS Sensor")
print("- Faster image capture")
print("- Lower power consumption")
print("- Cost-effective")
print("- Slightly higher image noise")

print("\nApplications")
print("CCD  : Medical Imaging, Astronomy")
print("CMOS : Smartphones, Surveillance, Autonomous Vehicles")
