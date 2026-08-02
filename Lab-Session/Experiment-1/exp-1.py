import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Screenshot 2026-07-20 133544.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite('gray_image.jpg', gray_img)
