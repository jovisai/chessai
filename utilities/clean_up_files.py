import os
from skimage.io import imread

folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_2'
op_images = []
for file in os.listdir(folder):
    img = imread(os.path.join(folder, file))
    sp = img.shape
    if sp[0] != 770:
        os.remove(os.path.join(folder, file))