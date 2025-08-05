# ------------------------------------------------------------- #
# LinkageNet                                                    #
# Utility functions for data pipeline                           #
#                                                               #
# ------------------------------------------------------------- #
# IVPG Lab                                                      #
# Author: Quang Huy Bui                                         #
# Modified date: August 2025                                    #
# ------------------------------------------------------------- #


import numpy as np

from config.config import MODEL_CONFIG

# Find intersection of a line segment with a rectangle (there can be 0 or 1 intersection)
def line_rec_intersection(point1, point2, img_height, img_width):
    if point2[0] < 0:
        x = 0
        y = point1[1] + (point2[1] - point1[1]) * (x - point1[0]) / (point2[0] - point1[0])
    elif point2[0] > img_width:
        x = img_width
        y = point1[1] + (point2[1] - point1[1]) * (x - point1[0]) / (point2[0] - point1[0])
    elif point2[1] < 0:
        y = 0
        x = point1[0] + (point2[0] - point1[0]) * (y - point1[1]) / (point2[1] - point1[1])
    elif point2[1] > img_height:
        y = img_height
        x = point1[0] + (point2[0] - point1[0]) * (y - point1[1]) / (point2[1] - point1[1])
    else:
        x, y = point2
    
    return np.array([x, y])



# Transform slot from two-point representation to four-point representation
# (x1, y1, x2, y2, o1x, o1y, o2x, o2y) --> (x1, y1, x2, y2, x3, y3, x4, y4)
def find_corners(slot, img_height, img_width, l_mean):
    junc1 = slot[0:2]
    ori1 = slot[4:6]
    point1 = junc1 + l_mean[int(slot[8])] * ori1
    junc4 = line_rec_intersection(junc1, point1, img_height, img_width)

    junc2 = slot[2:4]
    ori2  = slot[6:8]
    point2 = junc2 + l_mean[int(slot[8])] * ori2
    junc3 = line_rec_intersection(junc2, point2, img_height, img_width)

    return np.array([junc1[0], junc1[1], junc2[0], junc2[1], junc3[0], junc3[1], junc4[0], junc4[1]])



# Check if a cell lays inside a parking slot, used for low resolution feature map
# Return:
#   0: Not inside a parking slot
#   1: Inside a parking slot
#   -1: In ambiguos region, should be ignored
def inside_slot(point, slot, img_height, img_width, l_mean):
    corners = find_corners(slot, img_height, img_width, l_mean) / MODEL_CONFIG['low_feat_stride']

    def cross(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]
    
    p = [point[0] + 0.5, point[1] + 0.5]
    for i in range(4):
        p1 = [corners[2 * i], corners[2 * i + 1]]
        p2 = [corners[(2 * i + 2) % 8], corners[(2 * i + 3) % 8]]

        edge = (p2[0] - p1[0], p2[1] - p1[1])
        to_point = (p[0] - p1[0], p[1] - p1[1])

        if cross(edge, to_point) > 0:
            return 0
    return 1
