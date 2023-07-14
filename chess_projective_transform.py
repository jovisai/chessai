from skimage import transform
from skimage.io import imread
from skimage.transform import resize
import concurrent.futures
import matplotlib.pyplot as plt

import numpy as np
import os

import matplotlib
matplotlib.use('agg')

points = np.array([[ 40.68615777, 633.36571441],
    [225.00568365, 397.44747961],
    [389.24616159, 807.15132162],
    [514.68508139, 504.3149802 ]])

def process(file_path):
    plt.figure(figsize=(10,10))
    try:
        # print(file_path)
        cimage = imread(file_path)
        cimage = resize(cimage, (cimage.shape[0] // 4, cimage.shape[1] // 4),
                            anti_aliasing=True)
        h, w, _ = cimage.shape

        src = np.array([[0, 0], [0, w], [h, w], [h, 0]])
        dst = np.array([points[0], points[2], points[3], points[1]])

        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        warped = transform.warp(cimage, tform3, output_shape=(w, h))
        # flipped = np.flipud(warped)

        final = resize(warped, (w, w),
                            anti_aliasing=True)

        plt.figure(figsize=(10,10))
        plt.imshow(final)
        plt.axis("off")
        plt.savefig(
                "temp/output_frames_2/{0}".format(os.path.basename(file_path)),
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

for i in range(10):
    op_images = []
    for file in os.listdir('/home/leopard/development/jovis.ai/chessai/temp/output_frames_2'):
        op_images.append(file)

    unprocessed = []
    for file in os.listdir('/home/leopard/development/jovis.ai/chessai/temp/video_frames_2'):
        if file not in op_images:
            unprocessed.append(os.path.join('/home/leopard/development/jovis.ai/chessai/temp/video_frames_2', file))

    print(len(unprocessed))
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
            print(future.result())