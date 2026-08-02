import cv2
import numpy as np

# Read the image
image = cv2.imread("sample.jpg")

# Check if the image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Create a 5x5 kernel
kernel = np.ones((5, 5), np.uint8)

# Dilate the image
dilated = cv2.dilate(image, kernel, iterations=1)

# Display the original and dilated images
cv2.imshow("Original Image", image)
cv2.imshow("Dilated Image", dilated)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
