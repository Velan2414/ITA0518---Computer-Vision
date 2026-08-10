import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# --------------------------------------------------
# 1. Select Image
# --------------------------------------------------

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

print("Image selected successfully.")


# --------------------------------------------------
# 2. Rotate Image
# --------------------------------------------------

def rotate_image(img, angle):

    height, width = img.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        img,
        matrix,
        (width, height)
    )

    return rotated


# --------------------------------------------------
# 3. Add Gaussian Noise
# --------------------------------------------------

def add_noise(img, noise_level):

    # Generate Gaussian noise
    noise = np.random.normal(
        0,
        noise_level,
        img.shape
    )

    noisy = img.astype(np.float32) + noise

    # Keep pixel values between 0 and 255
    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(np.uint8)


# --------------------------------------------------
# 4. Create transformed images
# --------------------------------------------------

images = []

# Original
images.append(
    ("Original", image)
)

# Rotation only
rotated_30 = rotate_image(
    image,
    30
)

images.append(
    ("Rotation 30°", rotated_30)
)

# Rotation + low noise
rotated_30_noise_10 = add_noise(
    rotated_30,
    10
)

images.append(
    ("30° + Noise 10", rotated_30_noise_10)
)

# Rotation + medium noise
rotated_30_noise_25 = add_noise(
    rotated_30,
    25
)

images.append(
    ("30° + Noise 25", rotated_30_noise_25)
)

# Rotation + high noise
rotated_30_noise_50 = add_noise(
    rotated_30,
    50
)

images.append(
    ("30° + Noise 50", rotated_30_noise_50)
)


# --------------------------------------------------
# 5. Create ORB detector
# --------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=2000
)


# --------------------------------------------------
# 6. Detect features in original image
# --------------------------------------------------

gray_original = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

keypoints_original, descriptors_original = (
    orb.detectAndCompute(
        gray_original,
        None
    )
)

print("\nOriginal Features:",
      len(keypoints_original))


# --------------------------------------------------
# 7. BF Matcher
# --------------------------------------------------

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


# --------------------------------------------------
# 8. Perform Feature Matching
# --------------------------------------------------

results = []

for name, img in images:

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    # Original image does not need matching
    if name == "Original":

        results.append(
            (
                name,
                len(keypoints),
                0,
                None
            )
        )

        continue

    if descriptors is None:

        results.append(
            (
                name,
                len(keypoints),
                0,
                None
            )
        )

        continue

    # Match descriptors
    matches = bf.match(
        descriptors_original,
        descriptors
    )

    # Sort by distance
    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    # Select good matches
    good_matches = matches[:50]

    results.append(
        (
            name,
            len(keypoints),
            len(good_matches),
            good_matches
        )
    )


# --------------------------------------------------
# 9. Print Results
# --------------------------------------------------

print("\n==============================================")
print(" FEATURE MATCHING UNDER NOISE AND ROTATION")
print("==============================================")

print(
    f"{'Condition':<25}"
    f"{'Features':<12}"
    f"{'Good Matches':<15}"
)

print("-" * 52)

for result in results:

    name = result[0]
    feature_count = result[1]
    good_match_count = result[2]

    print(
        f"{name:<25}"
        f"{feature_count:<12}"
        f"{good_match_count:<15}"
    )


# --------------------------------------------------
# 10. Display Matching Results
# --------------------------------------------------

for name, img in images:

    if name == "Original":
        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    if descriptors is None:
        continue

    matches = bf.match(
        descriptors_original,
        descriptors
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = matches[:30]

    matched_image = cv2.drawMatches(
        image,
        keypoints_original,
        img,
        keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.imshow(
        cv2.cvtColor(
            matched_image,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"{name} - Good Matches: "
        f"{len(good_matches)}"
    )

    plt.axis("off")

    plt.show()


# --------------------------------------------------
# 11. Create comparison graph
# --------------------------------------------------

labels = [
    result[0]
    for result in results[1:]
]

match_counts = [
    result[2]
    for result in results[1:]
]

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    labels,
    match_counts,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Transformation Condition"
)

plt.ylabel(
    "Number of Good Matches"
)

plt.title(
    "Effect of Noise and Rotation on Feature Matching"
)

plt.xticks(
    rotation=20
)

plt.grid(True)

plt.tight_layout()

plt.show()


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

with open(
    "feature_matching_results.csv",
    "w"
) as file:

    file.write(
        "Condition,Features,Good Matches\n"
    )

    for result in results:

        file.write(
            f"{result[0]},"
            f"{result[1]},"
            f"{result[2]}\n"
        )

print(
    "\nResults saved as "
    "feature_matching_results.csv"
)
