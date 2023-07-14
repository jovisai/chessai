import skimage.io
import os

# Get the path to the video file.
video_file = "temp/VID_20230713_213108.mp4"

import cv2
import os

# Create a directory to store the images.
image_dir = "temp/video_frames_2"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

# Get the number of frames in the video.
cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Iterate over the frames in the video and save them as images.
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        continue
    image_name = "frame_{}.jpg".format(i)
    cv2.imwrite(os.path.join(image_dir, image_name), frame)

cap.release()