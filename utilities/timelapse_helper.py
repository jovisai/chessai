from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv
from skimage.transform import rescale, resize
import numpy as np
import os
from skimage.color import rgb2lab
from skimage.transform import rescale, resize
from skimage.io import imread
import shutil
import concurrent.futures

def locate_colors(lab_img, L_val_min, A_val_min, A_val_max, B_val_min, B_val_max, color):
    x, y, _ = lab_img.shape
    red_pixels = []
    for xi in range(x):
        for yi in range(y):
            L_val = lab_img[xi,yi][0] 
            A_val = lab_img[xi,yi][1] 
            B_val = lab_img[xi,yi][2]
            if L_val > L_val_min and A_val > A_val_min and A_val < A_val_max  and B_val > B_val_min and B_val < B_val_max:
                red_pixels.append([xi, yi])
    return red_pixels

def find_mean(image_path):
    image = imread(image_path)
    simage = resize(image, (image.shape[0] // 8, image.shape[1] // 8),
                           anti_aliasing=True)
    lab_img = rgb2lab(simage)
    red_pixles = locate_colors(lab_img, 30, 25, 100, 0, 100, [255, 0, 0])
    red_pixles = np.array(red_pixles)
    return red_pixles, simage, int(np.mean(red_pixles[:,1])), int(np.mean(red_pixles[:,0]))


folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_3_1/'
dst = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_3_3/'

already_processed = os.listdir(dst)
not_yet_processed = []
for file in os.listdir(folder):
    if file not in already_processed:
        not_yet_processed.append(file)

def process(filename):
    red_pixles, simage, x, y = find_mean(os.path.join(folder, filename))
    if y == 62:
        shutil.copyfile(os.path.join(folder, filename), os.path.join(dst, filename))
        return "found one: {0}".format(filename)
    return ""

with concurrent.futures.ThreadPoolExecutor() as pool:
    # Submit tasks to the thread pool.
    futures = [pool.submit(process, filename) for filename in not_yet_processed]

    # Get the results of the tasks.
    for future in futures:
        print(future.result())