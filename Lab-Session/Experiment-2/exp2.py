import cv2
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("img1.jpeg")

# Blur image
blur = cv2.GaussianBlur(image, (5,5), 0)

# Show image
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
