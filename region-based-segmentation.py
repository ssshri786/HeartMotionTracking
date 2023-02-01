#Samyak Shrimali, Dr. Rugyoni
#Heart Modeling, Region-Based Segmentation
#Notes: 
# The snake will converge to the boundary of the object in the image (the heart wall). 
# Need to adjust the parameters alpha, beta, and gamma to get the desired result.
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

# Loop through all the frames in the "dicom_images" directory
for i, image_path in enumerate(sorted(os.listdir("dicom_images"))):
    # Load the image
    img = cv2.imread(os.path.join("dicom_images", image_path), 0)

    # Implement the region-based segmentation algorithm
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[100:300, 100:300] = 255
    mask = np.dstack([mask] * 3)
    snake = cv2.activeContour(img, mask, alpha=0.015, beta=10, gamma=0.001)

    # Draw the snake on the image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.polylines(img, [np.int32(snake)], True, (0, 255, 0), 2)

    # Save the segmented frame
    segmented_frame_path = os.path.join("segmented_frames", "frame_{}.png".format(i))
    cv2.imwrite(segmented_frame_path, img)
