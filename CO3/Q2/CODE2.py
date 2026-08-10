import cv2
import matplotlib.pyplot as plt
import time
from tkinter import Tk, filedialog

# ---------------------------------------------
# 1. Select image
# ---------------------------------------------

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("All Files", "*.*")
    ]
)

if not image_path:
    print("No image selected.")
    exit()

image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to read image.")
    exit()

# Convert to RGB for displaying with matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------------------
# 2. Create blurred images
# ---------------------------------------------

blur_5 = cv2.GaussianBlur(image, (5, 5), 0)
blur_11 = cv2.GaussianBlur(image, (11, 11), 0)
blur_21 = cv2.GaussianBlur(image, (21, 21), 0)

images = [
    ("Original", image),
    ("Gaussian Blur 5x5", blur_5),
    ("Gaussian Blur 11x11", blur_11),
    ("Gaussian Blur 21x21", blur_21)
]

# ---------------------------------------------
# 3. ORB Feature Detection
# ---------------------------------------------

results = []

for name, img in images:

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Start timer
    start = time.perf_counter()

    # Detect features
    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    # End timer
    end = time.perf_counter()

    detection_time = (end - start) * 1000

    number_features = len(keypoints)

    # Draw detected features
    output = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    results.append(
        (
            name,
            number_features,
            detection_time,
            output
        )
    )

# ---------------------------------------------
# 4. Display feature detection results
# ---------------------------------------------

plt.figure(figsize=(14, 10))

for i, (name, count, detection_time, output) in enumerate(results):

    plt.subplot(2, 2, i + 1)

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(output_rgb)

    plt.title(
        f"{name}\n"
        f"Features: {count} | "
        f"Time: {detection_time:.2f} ms"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------
# 5. Print observation table
# ---------------------------------------------

print("\n==============================================")
print("       EFFECT OF BLURRING ON FEATURES")
print("==============================================")

print(
    f"{'Image':<25}"
    f"{'Features':<12}"
    f"{'Time (ms)':<12}"
)

print("-" * 50)

for name, count, detection_time, output in results:

    print(
        f"{name:<25}"
        f"{count:<12}"
        f"{detection_time:<12.2f}"
    )

# ---------------------------------------------
# 6. Extract data for graph
# ---------------------------------------------

labels = [r[0] for r in results]
feature_counts = [r[1] for r in results]
times = [r[2] for r in results]

# ---------------------------------------------
# 7. Feature count graph
# ---------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    labels,
    feature_counts,
    marker="o",
    linewidth=2
)

plt.xlabel("Blur Level")
plt.ylabel("Number of Detected Features")
plt.title("Effect of Blurring on Feature Detection")
plt.grid(True)

plt.show()

# ---------------------------------------------
# 8. Detection time graph
# ---------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    labels,
    times,
    marker="o",
    linewidth=2
)

plt.xlabel("Blur Level")
plt.ylabel("Detection Time (ms)")
plt.title("Effect of Blurring on Detection Time")
plt.grid(True)

plt.show()

# ---------------------------------------------
# 9. Save results
# ---------------------------------------------

with open("blur_feature_results.csv", "w") as file:

    file.write(
        "Blur Level,Detected Features,Detection Time (ms)\n"
    )

    for name, count, detection_time, output in results:

        file.write(
            f"{name},{count},{detection_time:.2f}\n"
        )

print("\nResults saved to:")
print("blur_feature_results.csv")
