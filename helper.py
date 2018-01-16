#######################################################
# This file contains several helper methods segmented
# into sections based on their functionality. Author
# of these functions: Ryan McCaffrey
#######################################################

import cv2

# ------------------------------------
# File reading functions
# ------------------------------------

IMAGE_PREFIX = 'img/FDDB-pics'

# for a given file that contains a list of names of images,
# populates a list of all of the images and returns it
def get_images_from_file_list(file_list):
    image_list = []
    for file in file_list:
        img = cv2.imread('{}/{}.jpg'.format(IMAGE_PREFIX, file))
        image_list.append(img)
    return image_list

# ------------------------------------
# Math functions
# ------------------------------------

# takes a float (or list of them) and rounds them to nearest int
def round_int(num):
    if type(num) is float:
        return int(round(num))
    else:
        return [int(round(x)) for x in num]

# From a given face label, which contains elliptical data:
# <major_axis_radius minor_axis_radius angle center_x center_y 1>,
# compute the bounding box for the face
def get_box_from_ellipse(major, minor, angle, h, k):
    # lambda functions for computing x and y from parametric equiations for arbitrarily rotated ellipse
    comp_x = lambda t, h, a, b, phi: h + a*math.cos(t)*math.cos(phi) - b*math.sin(t)*math.sin(phi)
    comp_y = lambda t, k, a, b, phi: k + b*math.sin(t)*math.cos(phi) + a*math.cos(t)*math.sin(phi)

    # before any computation done, check if angle is 0
    if angle == 0:
        return (h - minor/2, k - major/2, minor, major)

    radians = (angle * math.pi) / 180

    
    # take gradient of ellipse equations with respect to t and set to 0. Yields
    # 0 = dx/dt = -a*sin(t)*cos(phi) - b*cos(t)*sin(phi)
    # 0 = dy/dt =  b*cos(t)*cos(phi) - a*sin(t)*sin(phi)
    # and then solve for t
    tan_t_x = -1 * minor * math.tan(radians) / major
    tan_t_y = minor * (1/math.tan(radians)) / major
    arctan_x = math.atan(tan_t_x)
    arctan_y = math.atan(tan_t_y)
    
    # compute left and right of bounding box
    x_min, x_max = comp_x(arctan_x, h, minor, major, radians), comp_x(arctan_x + math.pi, h, minor, major, radians)
    if x_max < x_min:
        x_min, x_max = x_max, x_min

    # compute top and bottom of bounding box
    y_min, y_max = comp_y(arctan_y, k, minor, major, radians), comp_y(arctan_y + math.pi, k, minor, major, radians)
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    # return tuple (x_min, y_min, width, height)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

# Takes two bounding boxes and returns the intersection
# of their areas
def area_intersection(box_1, box_2):
    x1,y1,w1,h1 = box_1
    x2,y2,w2,h2 = box_2

    dx = min(x1+w1, x2+w2) - max(x1, x2)
    dy = min(y1+h1, y2+h2) - max(y1, y2)
    if dx >= 0 and dy >= 0:
        return dx*dy
    return 0

# Takes two bounding boxes and returns the union of their
# areas
def area_union(box_1, box_2):
    x1,y1,w1,h1 = box_1
    x2,y2,w2,h2 = box_2

    return w1*h1 + w2*h2 - area_intersection(box_1, box_2)

# computes the continuous score between two bounding boxes. The score is computed as 
# S = (area(box_1) intersection area(box_2)) / (area(box_1) union area(box_2)) 
# as seen in paper by Jain, Learned-Miller in paper 
# FDDB: A Benchmark for Face Detection in Unconstrained Settings (unpublished)
def continuous_score(box_1, box_2):
    if box_1 is None or box_2 is None:
        return 0
    return area_intersection(box_1, box_2)/area_union(box_1,box_2)

