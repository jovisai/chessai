import matplotlib.pyplot as plt
import cv2
import urllib
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab, rgb2hsv
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
import numpy as np
from skimage.util import img_as_float
import os
from skimage.transform import rescale, resize, downscale_local_mean
import concurrent.futures
import matplotlib
import shutil


matplotlib.use('agg')


def percent_skin(img_path):
    img=cv2.imread(img_path)
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

    w,h=global_mask.shape
    percent_skin = (np.sum(global_mask != 0)/(w * h)) * 100

    return percent_skin
def export_figure_matplotlib(arr, f_name, dpi=600, resize_fact=1):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr)
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    plt.close()

def draw_lines(url):
    try:
        percent = percent_skin(url)
        
        # original file
        cimage = imread(os.path.join(src_folder, os.path.basename(url)))

        lines = [[-816.4020821183715, 2243.046285815963, 0.3639702342662024],
        [-830.6919083474568, 2221.786432897199, 0.37388467948480464],
        [2534.670857939903, 972.9689830154904, -2.6050890646938014],
        [-801.4702690202839 ,2263.2822201123204 ,0.35411857253069795],
        [2528.078238781425, 945.2097220193025, -2.674621493926825],
        [-334.95211991801517, 1450.8370264652153, 0.2308681911255632],
        [1268.6374830035884 ,1336.8649657756464, -0.9489645667148797]]
        
        dpi = 600
        h, w, _ = cimage.shape
        plt.figure(figsize=(h/dpi, w/dpi))
        plt.xlim(0, w)
        plt.ylim(h, 0) 
        plt.axis('off')
        plt.imshow(cimage, cmap='gray')
        if percent < 1:
            for l in lines:
                plt.axline((l[0], l[1]), slope=l[2], c='r', lw=1)

        plt.savefig(
                "temp/output_frames_4_2/{0}".format(os.path.basename(url)),
                pad_inches=0,
                bbox_inches="tight",
                dpi=dpi*2, 
                format="jpg",
            )
        plt.close('all')
    except e:
        print("Exception", e)
        plt.close('all')

def process(filename):
    # detect hand
    percent = percent_skin(os.path.join(isrc_folder, filename))
    if percent > 1:
        shutil.copyfile(os.path.join(src_folder, filename), os.path.join(op_folder, filename))


src_folder = '/home/leopard/development/jovis.ai/chessai/temp/video_frames_4'
isrc_folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_4_1'
op_folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_4_2'

print("runnning iteration ", 0)
op_images = os.listdir(op_folder)

unprocessed = []
for file in os.listdir(isrc_folder):
    if file not in op_images:
        unprocessed.append(os.path.join(isrc_folder, file))

print("processing-->", len(unprocessed))
# with concurrent.futures.ThreadPoolExecutor() as pool:
#     # Submit tasks to the thread pool.
#     futures = [pool.submit(draw_lines, path) for path in unprocessed]

#     # Get the results of the tasks.
#     for future in futures:
#         future.result()
#         print("|", end =" ")

for path in unprocessed:
    draw_lines(path)