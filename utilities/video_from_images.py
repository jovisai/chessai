import cv2
import numpy as np
import os

def create_video_from_images(image_files, output_file, fps=36):
  """Creates a video from a list of image files.

  Args:
    image_files: A list of image files.
    output_file: The output video file.
    fps: The frames per second of the output video.
  """

  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  cap = cv2.VideoCapture(image_files[0])
  video_writer = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)),int(cap.get(4))))

  for image_file in image_files:
    image = cv2.imread(image_file)
    video_writer.write(image)

  video_writer.release()


files = os.listdir('/home/leopard/development/jovis.ai/chessai/temp/video_frames_2')
size = len(files)
print(size)
image_files = []
for i in range(3000):
   path = '/home/leopard/development/jovis.ai/chessai/temp/video_frames_2/frame_{0}.jpg'.format(i)
   if os.path.exists(path):
    image_files.append(path)

print(len(image_files))
if __name__ == "__main__":
  output_file = "temp/video_2.avi"
  create_video_from_images(image_files, output_file)