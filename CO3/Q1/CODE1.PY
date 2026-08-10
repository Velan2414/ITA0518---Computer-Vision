import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --------------------------------------------------
# 1. Load image
# --------------------------------------------------

image_path = "img.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    print("Make sure 'image.jpg' is in the same folder as this program.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --------------------------------------------------
# 2. Image information
# --------------------------------------------------

height, width = gray.shape
image_area = width * height

print("\n========== IMAGE INFORMATION ==========")
print("Width       :", width)
print("Height      :", height)
print("Image Area  :", image_area)

# --------------------------------------------------
# 3. Different feature limits
# --------------------------------------------------

feature_limits = [100, 500, 1000, 2000]

results = []

# --------------------------------------------------
# 4. ORB feature detection
# --------------------------------------------------

for max_features in feature_limits:

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)

    # Start timer
    start_time = time.perf_counter()

    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # End timer
    end_time = time.perf_counter()

    # Detection time in milliseconds
    detection_time = (end_time - start_time) * 1000

    # Number of detected features
    number_of_features = len(keypoints)

    # Feature density
    density = number_of_features / image_area

    # Features per 1000 pixels
    density_1000 = density * 1000

    # Store results
    results.append([
        max_features,
        number_of_features,
        density,
        density_1000,
        detection_time
    ])

    # --------------------------------------------------
    # Display detected features
    # --------------------------------------------------

    output_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(
        f"ORB Features: {number_of_features} | "
        f"Time: {detection_time:.2f} ms"
    )
    plt.axis("off")
    plt.show()

# --------------------------------------------------
# 5. Print results
# --------------------------------------------------

print("\n================ RESULTS ================")

print(
    f"{'Limit':<10}"
    f"{'Features':<12}"
    f"{'Density':<15}"
    f"{'Per 1000 px':<15}"
    f"{'Time (ms)':<12}"
)

print("-" * 64)

for result in results:

    limit = result[0]
    features = result[1]
    density = result[2]
    density_1000 = result[3]
    detection_time = result[4]

    print(
        f"{limit:<10}"
        f"{features:<12}"
        f"{density:<15.6f}"
        f"{density_1000:<15.6f}"
        f"{detection_time:<12.2f}"
    )

# --------------------------------------------------
# 6. Extract values for graphs
# --------------------------------------------------

features = [r[1] for r in results]
times = [r[4] for r in results]
densities = [r[3] for r in results]

# --------------------------------------------------
# 7. Graph 1 - Features vs Detection Time
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    features,
    times,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Detected Features")
plt.ylabel("Detection Time (ms)")
plt.title("Feature Density vs Computational Time")
plt.grid(True)

plt.show()

# --------------------------------------------------
# 8. Graph 2 - Feature Density
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    features,
    densities,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Detected Features")
plt.ylabel("Features per 1000 Pixels")
plt.title("Feature Density Analysis")
plt.grid(True)

plt.show()

# --------------------------------------------------
# 9. Save results to CSV
# --------------------------------------------------

with open("feature_density_results.csv", "w") as file:

    file.write(
        "Feature Limit,Detected Features,"
        "Density,Features per 1000 Pixels,Detection Time (ms)\n"
    )

    for result in results:

        file.write(
            f"{result[0]},"
            f"{result[1]},"
            f"{result[2]:.8f},"
            f"{result[3]:.6f},"
            f"{result[4]:.2f}\n"
        )

print("\nResults saved to:")
print("feature_density_results.csv")
