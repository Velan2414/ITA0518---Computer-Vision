import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import Tk, filedialog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. SELECT IMAGE
# ============================================================

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

print("\nImage loaded successfully.")


# ============================================================
# 2. CREATE TRANSFORMED IMAGE
# ============================================================

height, width = image.shape[:2]

center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    25,
    1.0
)

rotated_image = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)


# ============================================================
# 3. PREPROCESSING
# ============================================================

start_preprocessing = time.perf_counter()

# Resize
image_resized = cv2.resize(
    image,
    (800, 600)
)

rotated_resized = cv2.resize(
    rotated_image,
    (800, 600)
)

# Convert to grayscale
gray_original = cv2.cvtColor(
    image_resized,
    cv2.COLOR_BGR2GRAY
)

gray_rotated = cv2.cvtColor(
    rotated_resized,
    cv2.COLOR_BGR2GRAY
)

# Gaussian blur
gray_original = cv2.GaussianBlur(
    gray_original,
    (5, 5),
    0
)

gray_rotated = cv2.GaussianBlur(
    gray_rotated,
    (5, 5),
    0
)

end_preprocessing = time.perf_counter()

preprocessing_time = (
    end_preprocessing -
    start_preprocessing
) * 1000


# ============================================================
# 4. DISPLAY PREPROCESSING RESULT
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Original Image")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    gray_original,
    cmap="gray"
)

plt.title("Preprocessed Image")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 5. ORB FEATURE DETECTION
# ============================================================

orb = cv2.ORB_create(
    nfeatures=2000
)

start_detection = time.perf_counter()

keypoints_original, descriptors_original = (
    orb.detectAndCompute(
        gray_original,
        None
    )
)

keypoints_rotated, descriptors_rotated = (
    orb.detectAndCompute(
        gray_rotated,
        None
    )
)

end_detection = time.perf_counter()

detection_time = (
    end_detection -
    start_detection
) * 1000


print("\n====================================")
print("FEATURE DETECTION")
print("====================================")

print(
    "Original features :",
    len(keypoints_original)
)

print(
    "Rotated features  :",
    len(keypoints_rotated)
)

print(
    "Detection time    :",
    round(detection_time, 2),
    "ms"
)


# ============================================================
# 6. DISPLAY DETECTED FEATURES
# ============================================================

feature_image = cv2.drawKeypoints(
    image_resized,
    keypoints_original,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 6))

plt.imshow(
    cv2.cvtColor(
        feature_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"ORB Detected Features: "
    f"{len(keypoints_original)}"
)

plt.axis("off")
plt.show()


# ============================================================
# 7. FEATURE MATCHING
# ============================================================

start_matching = time.perf_counter()

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

matches = bf.match(
    descriptors_original,
    descriptors_rotated
)

matches = sorted(
    matches,
    key=lambda x: x.distance
)

# Select best 50 matches
good_matches = matches[:50]

end_matching = time.perf_counter()

matching_time = (
    end_matching -
    start_matching
) * 1000


print("\n====================================")
print("FEATURE MATCHING")
print("====================================")

print(
    "Total matches :",
    len(matches)
)

print(
    "Good matches  :",
    len(good_matches)
)

print(
    "Matching time :",
    round(matching_time, 2),
    "ms"
)


# ============================================================
# 8. DISPLAY MATCHING RESULT
# ============================================================

matched_image = cv2.drawMatches(
    image_resized,
    keypoints_original,
    rotated_resized,
    keypoints_rotated,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(15, 7))

plt.imshow(
    cv2.cvtColor(
        matched_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"Feature Matching - "
    f"{len(good_matches)} Best Matches"
)

plt.axis("off")
plt.show()


# ============================================================
# 9. PREPARE DESCRIPTORS FOR PCA
# ============================================================

# Combine descriptors from both images

combined_descriptors = np.vstack(
    (
        descriptors_original,
        descriptors_rotated
    )
)

print("\n====================================")
print("DESCRIPTOR INFORMATION")
print("====================================")

print(
    "Original descriptor shape:",
    descriptors_original.shape
)

print(
    "Rotated descriptor shape :",
    descriptors_rotated.shape
)

print(
    "Combined descriptor shape:",
    combined_descriptors.shape
)


# ============================================================
# 10. STANDARDIZE DESCRIPTORS
# ============================================================

scaler = StandardScaler()

scaled_descriptors = scaler.fit_transform(
    combined_descriptors.astype(np.float32)
)


# ============================================================
# 11. APPLY PCA
# ============================================================

start_pca = time.perf_counter()

pca = PCA(
    n_components=2
)

pca_result = pca.fit_transform(
    scaled_descriptors
)

end_pca = time.perf_counter()

pca_time = (
    end_pca -
    start_pca
) * 1000


# ============================================================
# 12. PCA INFORMATION
# ============================================================

print("\n====================================")
print("PCA DIMENSIONALITY REDUCTION")
print("====================================")

print(
    "Original dimensions :",
    combined_descriptors.shape[1]
)

print(
    "Reduced dimensions  :",
    pca_result.shape[1]
)

print(
    "PC1 variance        :",
    round(
        pca.explained_variance_ratio_[0] * 100,
        2
    ),
    "%"
)

print(
    "PC2 variance        :",
    round(
        pca.explained_variance_ratio_[1] * 100,
        2
    ),
    "%"
)

print(
    "Total variance      :",
    round(
        sum(pca.explained_variance_ratio_) * 100,
        2
    ),
    "%"
)

print(
    "PCA processing time :",
    round(pca_time, 2),
    "ms"
)


# ============================================================
# 13. PCA VISUALIZATION
# ============================================================

number_original = len(
    descriptors_original
)

original_pca = pca_result[
    :number_original
]

rotated_pca = pca_result[
    number_original:
]

plt.figure(figsize=(10, 7))

plt.scatter(
    original_pca[:, 0],
    original_pca[:, 1],
    label="Original Image"
)

plt.scatter(
    rotated_pca[:, 0],
    rotated_pca[:, 1],
    label="Rotated Image"
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title(
    "PCA Visualization of Feature Descriptors"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 14. FINAL PERFORMANCE SUMMARY
# ============================================================

total_time = (
    preprocessing_time +
    detection_time +
    matching_time +
    pca_time
)

print("\n====================================")
print("FINAL SYSTEM PERFORMANCE")
print("====================================")

print(
    "Preprocessing time :",
    round(preprocessing_time, 2),
    "ms"
)

print(
    "Feature detection  :",
    round(detection_time, 2),
    "ms"
)

print(
    "Feature matching   :",
    round(matching_time, 2),
    "ms"
)

print(
    "PCA processing     :",
    round(pca_time, 2),
    "ms"
)

print(
    "Total pipeline time:",
    round(total_time, 2),
    "ms"
)

print("\nEnd-to-end vision pipeline completed successfully.")
