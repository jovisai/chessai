import cv2
import os
import math
import concurrent.futures
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use('agg')


def find_intersection_point(fp_x0, fp_y0, slope1, sp_x0, sp_y0, slope2):
    if (slope1 - slope2) == 0:
        return []

    # Calculate the intersection point coordinates
    x_intersect = (sp_y0 - fp_y0 + slope1 * fp_x0 - slope2 * sp_x0) / (slope1 - slope2)
    y_intersect = slope1 * (x_intersect - fp_x0) + fp_y0

    if x_intersect < 0 or y_intersect < 0 or x_intersect > 4000 or y_intersect > 4000:
        return []

    angle_of_intersection = math.degrees(
        math.atan((slope1 - slope2) / (1 + slope1 * slope2))
    )

    if angle_of_intersection < 45 and angle_of_intersection > -45:
        return []

    # Intersection point coordinates
    intersection_point = [x_intersect, y_intersect]

    return intersection_point


def segment_chess(file_path):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = image.copy()
    
    # convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Perform Otsu threshold on the A-channel 
    th = cv2.threshold(lab[:,:,1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    result = cv2.bitwise_and(result, result, mask=th)

    image_gray = rgb2gray(image_rgb)

    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(rgb2gray(result), theta=tested_angles)

    lines = []
    all_points_and_slopes = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        lines.append([x0, y0, angle])
        # ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))
        # ax[2].scatter(x0, y0)
        slope = np.tan(angle + np.pi / 2)
        all_points_and_slopes.append([x0, y0, slope])

    # find intersection points
    intersection_points = []
    for i in range(len(all_points_and_slopes)):
        for j in range(1, len(all_points_and_slopes)):
            p1 = all_points_and_slopes[i]
            p2 = all_points_and_slopes[j]

            ip = find_intersection_point(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
            if ip:
                intersection_points.append(ip)

    intersection_points = np.array(intersection_points)

    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(
        intersection_points
    )

    points = kmeans.cluster_centers_
    hull = ConvexHull(points)

    plt.figure(figsize=(11, 20))
    plt.imshow(image_gray, cmap="gray")
    plt.plot(points[:, 0], points[:, 1], "o", color="r")
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color="r")
    plt.axis("off")
    plt.savefig(
        "output/{0}".format(os.path.basename(file_path)),
        pad_inches=0,
        bbox_inches="tight",
        format="jpg",
    )


folder_path = "video_frames"
images = []
for file in os.listdir(folder_path):
    images.append(os.path.join(folder_path, file))

with concurrent.futures.ThreadPoolExecutor() as pool:
    # Submit tasks to the thread pool.
    futures = [pool.submit(segment_chess, path) for path in images]

    # Get the results of the tasks.
    for future in futures:
        print(future.result())
