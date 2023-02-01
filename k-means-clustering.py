#Samyak Shrimali, Dr. Rugyoni
#Heart Modeling, Region-Based Segmentation
#Notes: 
# Need to adjust the k to get the desired result and cluster heart wall.
# Can also try to change the color densities in each frame to highlight changes which can help with segmentation.

import pydicom
import numpy as np
import os

# Load a DICOM video
video = pydicom.dcmread("sample_dicom_video.dcm")

# Get the video data as a 3D numpy array
frames = np.stack([s.pixel_array for s in video.SequenceOfUltrasoundRegions], axis=-1)

# Create a directory to save the images
if not os.path.exists("dicom_images"):
    os.makedirs("dicom_images")

# Save each frame as an image
for i, frame in enumerate(frames):
    image_path = os.path.join("dicom_images", "frame_{}.png".format(i))
    cv2.imwrite(image_path, frame)

import cv2
import numpy as np
import os

# Create a directory to save the segmented frames
if not os.path.exists("segmented_frames"):
    os.makedirs("segmented_frames")

# Define the number of clusters
K = 2

# Loop through all the frames in the "dicom_images" directory
for i, image_path in enumerate(sorted(os.listdir("dicom_images"))):
    # Load the image
    img = cv2.imread(os.path.join("dicom_images", image_path))

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, 3).astype(np.float32)

    # Perform K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Create a mask for each cluster
    masks = [labels == k for k in range(K)]

    # Draw the clusters on the image
    colors = [(0, 0, 255), (0, 255, 0)]
    img = np.zeros_like(img)
    for mask, color in zip(masks, colors):
        img[mask.reshape(img.shape[:2])] = color

    # Save the segmented frame
    segmented_frame_path = os.path.join("segmented_frames", "frame_{}.png".format(i))
    cv2.imwrite(segmented_frame_path, img)
