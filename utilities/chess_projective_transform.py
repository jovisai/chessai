from skimage import transform
from skimage.io import imread
from skimage.transform import resize
import concurrent.futures
import matplotlib.pyplot as plt

import numpy as np
import os
import sys

import matplotlib
matplotlib.use('agg')

points = np.array([[ 856.94409294, 1727.5474053 ],
       [1696.4398573 , 3156.1802808 ],
       [2128.38043592, 2021.08161393],
       [   4.09110518, 2536.8746713 ]])
points = points[points[:, 0].argsort()]

def process(file_path):
    plt.figure(figsize=(10,10))
    try:
        # print(file_path)
        cimage = imread(file_path)
        # cimage = resize(cimage, (cimage.shape[0] // 4, cimage.shape[1] // 4),
        #                     anti_aliasing=True)
        h, w, _ = cimage.shape

        src = np.array([[0, 0], [0, w], [h, w], [h, 0]])
        dst = np.array([points[0], points[2], points[3], points[1]])

        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = transform.warp(cimage, tform3, output_shape=(w, h))
        # flipped = np.flipud(warped)

        final = resize(warped, (w, w),
                            anti_aliasing=True)
        final = transform.rotate(final, 90, resize=False, mode='constant', cval=0)

        plt.figure(figsize=(10,10))
        plt.imshow(final)
        plt.axis("off")
        plt.savefig(
                "temp/output_frames_4/{0}".format(os.path.basename(file_path)),
                pad_inches=0,
                bbox_inches="tight",
                format="jpg",
            )
        plt.close('all')
    except:
        plt.close('all')
        print("failed", file_path)
        # os.remove(file_path)
    return

src_folder = '/home/leopard/development/jovis.ai/chessai/temp/video_frames_4'
op_folder = '/home/leopard/development/jovis.ai/chessai/temp/output_frames_4'
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
    # for file in os.listdir('/home/leopard/development/jovis.ai/chessai/temp/video_frames_2'):
    #     if file not in op_images:
    #         #print(file)
    #         # pass
    #         process(os.path.join('/home/leopard/development/jovis.ai/chessai/temp/video_frames_2', file))
    # unprocessed = np.array(unprocessed)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        # Submit tasks to the thread pool.
        futures = [pool.submit(process, path) for path in unprocessed]

        # Get the results of the tasks.
        for future in futures:
            future.result()
            print("|", end =" ")