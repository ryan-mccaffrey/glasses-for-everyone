# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import src.facenet.detect_face
import cv2
import matplotlib.pyplot as plt
import math
import pickle
import dlib

# ============================================
# Global variables
# ============================================
IMAGE_PREFIX = 'img/FDDB-pics'
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709 # scale factor
face_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_frontalface_default.xml')
dlib_face_detector = dlib.get_frontal_face_detector()


# ============================================
# Face detection methods
# ============================================

# Uses the HOG face detection algorithm internal in the dlib library
def hog_face_detect(image):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = dlib_face_detector(gray, 1)
    return rects

# Acknowledgement: much of this code was taken from the blog of Charles Jekel, who explains
# how to use FaceNet to detect faces here: http://jekel.me/2017/How-to-detect-faces-using-facenet/
def cnn_face_detect(image):
    # Configuring facenet in facenet/src/compare.py
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = src.facenet.detect_face.create_mtcnn(sess, None)
        
        # run detect_face from the facenet library
        bounding_boxes, _ = src.facenet.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

        # for each face detection, compute bounding box and add as tuple
        face_detections = []
        for (x1, y1, x2, y2, acc) in bounding_boxes:
            w = x2 - x1
            h = y2 - y1
            face_detections.append((x1, y1, w, h))
            
        return face_detections

def haar_face_detect(image, scaleFactor, minNeighbors, use_grayscale=True, cascade=None):
    # convert to grayscale if needed
    if use_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not cascade:
        return face_cascade.detectMultiScale(image, scaleFactor, minNeighbors)
    else:
        return cascade.detectMultiScale(image, scaleFactor, minNeighbors)


# ============================================
# Helper functions
# ============================================

# for a given fold file that contains list of images in the fold,
# populates a list of all of the images in the fold and returns it
def get_image_list_from_file(file_name):
    image_list = []
    with open(file_name, 'r') as f:
        file_list = [x.rstrip() for x in f.readlines()]
        for file in file_list:
            img = cv2.imread('{}/{}.jpg'.format(IMAGE_PREFIX, file))
            image_list.append(img)
    return image_list

# From a given face label, which contains elliptical data:
# <major_axis_radius minor_axis_radius angle center_x center_y 1>,
# compute the bounding box for the face
def get_box_from_label(major, minor, angle, h, k):
    # lambda functions for computing x and y from parametric equiations for arbitrarily rotated ellipse
    comp_x = lambda t, h, a, b, phi: h + a*math.cos(t)*math.cos(phi) - b*math.sin(t)*math.sin(phi)
    comp_y = lambda t, k, a, b, phi: k + b*math.sin(t)*math.cos(phi) + a*math.cos(t)*math.sin(phi)

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

# For a given fold number [1-10], retrieve a nested list of bounding boxes for faces for each image
# in the fold. Ex data: [[img1_face1, img1_face2], [img2_face1], ...] where each face bounding box
# is a tuple of (x, y, width, height)
def retrieve_face_list(fold_num):
    assert fold_num > 0 and fold_num <= 10

    fold_file = 'img/FDDB-folds/FDDB-fold-{:02}-ellipseList.txt'.format(fold_num)
    rectangle_file = 'img/FDDB-folds/FDDB-fold-{:02}-rectangleList.pkl'.format(fold_num)

    # If this list has already been created, can load it from a pickle file
    if os.path.exists(rectangle_file):
        print("loading from pickle")
        with open(rectangle_file, 'rb') as f:
            face_list = pickle.load(f)
    else:
        face_list = []
        count, face_count = 0, 0
        with open(fold_file, 'r') as f:
            file_name = f.readline().rstrip()
            while file_name:
                num_faces = int(f.readline().rstrip())
                count += 1
                face_count += num_faces
                
                # iterates over each of the faces in image
                faces = []
                for i in range(num_faces):
                    major, minor, angle, h, k, _ = map(float, f.readline().rstrip().split())
                    faces.append(get_box_from_label(major, minor, angle, h, k))
                face_list.append(faces)

                # go to next file
                file_name = f.readline().rstrip()

        print('num images: {}, total num faces: {}'.format(count, face_count))
        with open(rectangle_file, 'wb') as w:
            pickle.dump(face_list, w)

    return face_list

# ============================================
# Testing methods
# ============================================

def test_haar(face_images, face_labels):
    total_faces, num_correct = 0, 0
    count = 0
    for image, label_set in zip(face_images, face_labels):
        count += 1
        print('image num: {}'.format(count))
        rows, cols, _ = image.shape
        
        # use predictor
        # predictions = haar_face_detect(image, 1.3, 5)
        predictions = cnn_face_detect(image)

        # sort labels by their centers
        label_set = sorted(label_set, key=lambda label: (label[0] + label[2]/2, label[1] + label[3]/2))
        predictions = sorted(predictions, key=lambda label: (label[0] + label[2]/2, label[1] + label[3]/2))

        total_faces += len(label_set)
        for i in range(len(label_set)):
        # for label, prediction in zip(label_set, predictions):
            if i >= len(predictions):
                break

            x_l, y_l, w_l, h_l = label_set[i]
            center_lx, center_ly = x_l + w_l/2, y_l + h_l/2
            x_p, y_p, w_p, h_p = predictions[i]
            center_px, center_py = x_p + w_p/2, y_p + h_p/2

            if (abs(center_lx - center_px) < .1*cols and abs(center_ly - center_py) < .1*rows):
                num_correct += 1



    print("found {} out of {} faces".format(num_correct, total_faces))
    print("accuracy: {}".format(num_correct/total_faces))

# The main method is used to compare the accuracies of the FaceNet detector and Haar Cascade detector
# 
def main():
    fold_num = 2
    img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
    face_images = get_image_list_from_file(img_list_file)
    face_labels = retrieve_face_list(fold_num)

    test_haar(face_images, face_labels)

    # haar_count, cnn_count, hog_count = 0, 0, 0
    # for image, label_set in zip(face_images, face_labels):
    # pass

    # label_set = [(2,2,6,6), (0,0,2,2), (3,3,2,2)]
    # label_set = sorted(label_set, key=lambda label: (label[0] + label[2]/2, label[1] + label[3]/2))
    # print(label_set)
    # print(5/2)




if __name__ == "__main__":
    main()
