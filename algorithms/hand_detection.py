import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, rgb2hsv
import os
from skimage.transform import rescale, resize
import concurrent.futures
import cv2
import numpy as np

import matplotlib
matplotlib.use('agg')

plt.figure(figsize=(10,10))


def detect_hand(image_path):
    try:
        img=cv2.imread(image_path)

        #converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #skin color range for hsv color space 
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


        HSV_result = cv2.bitwise_not(HSV_mask)
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        global_result=cv2.bitwise_not(global_mask)

        
        plt.axis("off")
        plt.imshow(imread(image_path))
        w,h=global_mask.shape
        percent_skin = (np.sum(global_mask != 0)/(w * h)) * 100
        if int(percent_skin) > 1:
            plt.text(100, 100, 'Hand Detected!', dict(size=24), c='r')
        plt.savefig(
                    "/home/leopard/development/jovis.ai/chessai/temp/output_frames_4_1/{0}".format(os.path.basename(image_path)),
                    pad_inches=0,
                    bbox_inches="tight",
                    format="jpg",
                )
        plt.close('all')
    except:
        plt.close('all')
        # print("failed", image_path)


def filter_color(cimage, L_val_min, A_val_min, A_val_max, B_val_min, B_val_max):
     # convert the image from RGB to LAB
    lab_img = rgb2lab(cimage)
    filtered_image = np.copy(cimage)
    x,y, _ = cimage.shape
    total_pixels = x * y
    colored_pixels = 0
    for xi in range(x):
        for yi in range(y):
            L_val = lab_img[xi,yi][0] 
            A_val = lab_img[xi,yi][1] 
            B_val = lab_img[xi,yi][2]
            if L_val > L_val_min and A_val > A_val_min and A_val < A_val_max  and B_val > B_val_min and B_val < B_val_max:
                colored_pixels = colored_pixels + 1
                
    color_percentage = colored_pixels*100/total_pixels
    return filtered_image, color_percentage


src_folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_4'
op_folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_4_1'
for i in range(10):
    print("runnning iteration ", i)
    op_images = []
    for file in os.listdir(op_folder):
        op_images.append(file)

    unprocessed = []
    for file in os.listdir(src_folder):
        if file not in op_images:
            unprocessed.append(os.path.join(src_folder, file))

    print("processing-->", len(unprocessed))
    with concurrent.futures.ThreadPoolExecutor() as pool:
        # Submit tasks to the thread pool.
        futures = [pool.submit(detect_hand, path) for path in unprocessed]

        # Get the results of the tasks.
        for future in futures:
            future.result()
            print("|", end =" ")