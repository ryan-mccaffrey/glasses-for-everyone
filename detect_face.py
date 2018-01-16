from datetime import datetime
from scipy import misc
import tensorflow as tf
import os
import src.facenet.detect_face
import cv2
import matplotlib.pyplot as plt
from helper import get_images_from_file_list, get_box_from_ellipse
import math
import pickle
import dlib

# ============================================
# Global variables
# ============================================

AVG_FACE_HEIGHT = 142.58539351061276
AVG_FACE_WIDTH = 94.11600875170973

# CNN global vars
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [0.5, 0.6, 0.7]  # three steps's threshold
factor = 0.800 # scale factor

# Haar and Dlib global vars
face_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_frontalface_default.xml')
dlib_face_detector = dlib.get_frontal_face_detector()


# ============================================
# Face detection methods
# ============================================

# For a given image, uses the dlib face detection algorithm to predict
# all of the faces present in the image. The algorithm used is based on 
# a 29-layer ResNet network architecture. Returns a list of dlib.rectangle
# objects
def dlib_face_detect(image, upscale=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = dlib_face_detector(gray, upscale)
    return rects

# For a given image, uses the FaceNet CNN detector to predict all of the faces
# present in the given image. Returns a list of bounding boxes (x,y,w,h) of the
# faces. This code was largely borrowed from the blog of Charles Jekel, found here:
# http://jekel.me/2017/How-to-detect-faces-using-facenet/
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
            # skip detections with < 60% confidence
            if acc < .6:
                continue

            w = x2 - x1
            h = y2 - y1
            face_detections.append((x1, y1, w, h))
            
        return face_detections

# For a given image, use the Haar Cascade detector provided by OpenCV to detect
# all of the faces present in the given image. Uses the parameters scale_factor and
# min_neighbors. Returns a list of bounding boxes (x,y,w,h) of the faces
def haar_face_detect(image, scale_factor, min_neighbors, use_grayscale=True, cascade=None):
    if use_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Can provide a different cascade type if desired. Cascades found in src/haarcascades
    if not cascade:
        return face_cascade.detectMultiScale(image, scale_factor, min_neighbors)
    else:
        return cascade.detectMultiScale(image, scale_factor, min_neighbors)


# ============================================
# Helper functions
# ============================================

# For a given fold number [1-10], retrieve a nested list of bounding boxes for faces for each image
# in the fold. Ex data: [[img1_face1, img1_face2], [img2_face1], ...] where each face bounding box
# is a tuple of (x, y, width, height)
def retrieve_face_list(fold_num):
    assert fold_num > 0 and fold_num <= 10

    fold_file = 'img/FDDB-folds/FDDB-fold-{:02}-ellipseList.txt'.format(fold_num)
    rectangle_file = 'img/FDDB-folds/FDDB-fold-{:02}-rectangleList.pkl'.format(fold_num)

    # If this list has already been created, can load it from a pickle file
    if os.path.exists(rectangle_file):
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
                    faces.append(get_box_from_ellipse(major, minor, angle, h, k))
                face_list.append(faces)

                # go to next file
                file_name = f.readline().rstrip()

        print('num images: {}, total num faces: {}'.format(count, face_count))
        with open(rectangle_file, 'wb') as w:
            pickle.dump(face_list, w)

    return face_list

def retrieve_manual_face_labels(fold_num, file_names):
    file_list = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
    rectangle_file = 'img/manual/face_labels.pkl'

    if os.path.exists(rectangle_file):
        print("loading from pickle")
        with open(rectangle_file, 'rb') as f:
            face_list = pickle.load(f)
            return face_list

    with open(file_list, 'r') as f:
        file_list = [x.rstrip() for x in f.readlines()]

    rectangles = retrieve_face_list(fold_num)
    face_list = []
    for f in file_names:
        for i, file in enumerate(file_list):
            if f == file:
                face_list.append(rectangles[i])
                break

    with open(rectangle_file, 'wb') as f:
        pickle.dump(face_list, f)

    return face_list


# ============================================
# Testing methods
# ============================================

# TODO: replace with a max flow?
def compute_accuracy(labels, predictions):
    faces_found, false_pos = 0, 0
    for prediction in predictions:
        if type(prediction) == dlib.dlib.rectangle:
                x_p, y_p, w_p, h_p = prediction.left(), prediction.top(), prediction.right()-prediction.left(), prediction.bottom()-prediction.top()
        else:
            x_p, y_p, w_p, h_p = prediction
        center_px, center_py = x_p + w_p/2, y_p + h_p/2

        found_one = False
        for label in labels:
            x_l, y_l, w_l, h_l = label
            center_lx, center_ly = x_l + w_l/2, y_l + h_l/2

            if (abs(center_lx - center_px) < .4*w_l and abs(center_ly - center_py) < .4*h_l
                and .5*w_l < w_p and w_p < 1.5*w_l and .5*h_l < h_p and h_p < 1.5*h_l):
                # num_correct += 1
                faces_found += 1
                found_one = True
                break

        if found_one is False:
            false_pos += 1

    if faces_found > len(labels):
        diff = faces_found - len(labels)
        false_pos += diff
        faces_found = len(labels)

    return faces_found, len(labels), false_pos


def write_detections(fold_num, file_names, face_images, face_labels):
    directory = 'pred/facenet/{:03}-{}{}{}'.format(int(factor*1000), int(threshold[0]*10), int(threshold[1]*10), int(threshold[2]*10))
    file = directory + '/fold-{}.pkl'.format(fold_num)

    print(file)
    # return

    if os.path.exists(file):
        print('file {} already exists'.format(file))
        return

    if not os.path.exists(directory):
        os.makedirs(directory)

    all_predictions = []
    for image in face_images:
        predictions = cnn_face_detect(image)
        all_predictions.append(predictions)

    with open(file, 'wb') as f:
        pickle.dump(all_predictions, f)

def test_detection(fold_num, file_names, face_images, face_labels):
    total_faces, total_num_correct, total_false_pos = 0, 0, 0
    count = 0
    for image, label_set in zip(face_images, face_labels):
        file = file_names[count]
        count += 1
        
        # choose detector
        # predictions = haar_face_detect(image, 1.25, 5)
        predictions = cnn_face_detect(image)
        # predictions = dlib_face_detect(image)

        num_correct, num_faces, false_pos = compute_accuracy(label_set, predictions)

        total_num_correct += num_correct
        total_faces += num_faces
        total_false_pos += false_pos

    # print("found {} out of {} faces in ".format(total_num_correct, total_faces))
    # print("accuracy: {}".format(num_correct/total_faces))
    return total_num_correct, total_faces, total_false_pos

def test_dlib_detection(fold_num, file_names, face_images, face_labels, upscale):
    total_faces, total_num_correct, total_false_pos = 0, 0, 0
    for image, label_set in zip(face_images, face_labels):
        predictions = dlib_face_detect(image, upscale=upscale)
        num_correct, num_faces, false_pos = compute_accuracy(label_set, predictions)
        total_faces += num_faces
        total_num_correct += num_correct
        total_false_pos += false_pos
    return total_num_correct, total_faces, total_false_pos

def test_haar_detection(fold_num, file_names, face_images, face_labels, scale_factor, min_neighbors):
    total_faces, total_num_correct, total_false_pos = 0, 0, 0
    for image, label_set in zip(face_images, face_labels):
        predictions = haar_face_detect(image, scale_factor, min_neighbors)
        num_correct, num_faces, false_pos = compute_accuracy(label_set, predictions)
        total_faces += num_faces
        total_num_correct += num_correct
        total_false_pos += false_pos
    return total_num_correct, total_faces, total_false_pos

def test_cnn_detection(fold_num, file_names, face_images, face_labels):
    directory = 'predictions/facenet/{:03}-{}{}{}'.format(int(factor*1000), int(threshold[0]*10), int(threshold[1]*10), int(threshold[2]*10))
    pkl_file = directory + '/fold-{}.pkl'.format(fold_num)
    total_faces, total_num_correct, total_false_pos = 0, 0, 0

    if os.path.exists(pkl_file):
        print('found file, loading')
        with open(pkl_file, 'rb') as f:
            fold_predictions = pickle.load(f)
        
        # iterates over each image in the fold
        for face_detections, labels in zip(fold_predictions, face_labels):
            num_correct, num_faces, false_pos = compute_accuracy(labels, face_detections)
            total_num_correct += num_correct
            total_faces += num_faces
            total_false_pos += false_pos

        return total_num_correct, total_faces, total_false_pos

    # predictions do not already exist for the fold, so make them and then write them
    count = 0
    fold_predictions = []
    for image, label_set in zip(face_images, face_labels):
        file = file_names[count]
        count += 1
        
        predictions = cnn_face_detect(image)
        fold_predictions.append(predictions)
        num_correct, num_faces, false_pos = compute_accuracy(label_set, predictions)
        total_num_correct += num_correct
        total_faces += num_faces
        total_false_pos += false_pos    

    with open(pkl_file, 'wb') as f:
        pickle.dump(fold_predictions, f)

    return total_num_correct, total_faces, total_false_pos


def test_on_one_image(file_names, face_labels):
    name = '2002/08/05/big/img_3688'
    img = cv2.imread('img/FDDB-pics/{}.jpg'.format(name))

    index = -1
    for i, file in enumerate(file_names):
        if name in file:
            index = i
            break

    print('found file at index {}'.format(i))

    # faces = cnn_face_detect(img)
    faces = haar_face_detect(img, 1.3, 4)
    label_set = face_labels[i]
    print("detections: (x,y,w,h)")

    # for i in range(len(label_set)):
    for i, prediction in enumerate(faces):
        print("*************** prediction {} *************".format(i))
        x_p, y_p, w_p, h_p = prediction
        print(x_p,y_p,w_p,h_p)
        cv2.rectangle(img,(int(x_p),int(y_p)),(int(x_p+w_p),int(y_p+h_p)),(255,0,0),2)
        center_px, center_py = x_p + w_p/2, y_p + h_p/2
        
        found_one = False
        for label in label_set:
            x_l, y_l, w_l, h_l = label
            print(x_l, y_l, w_l, h_l)
            center_lx, center_ly = x_l + w_l/2, y_l + h_l/2

            print(abs(center_lx - center_px) < .3*w_l)
            print(abs(center_ly - center_py) < .3*h_l)
            print(.5*w_l < w_p and w_p < 1.5*w_l)
            print(.5*h_l < h_p and h_p < 1.5*h_l)
            print("//////////////////")
            if (abs(center_lx - center_px) < .3*w_l and abs(center_ly - center_py) < .3*h_l
                and .5*w_l < w_p and w_p < 1.5*w_l and .5*h_l < h_p and h_p < 1.5*h_l):
                # num_correct += 1
                # faces_found_in_img += 1
                found_one = True
                break

        if found_one is False:
            print('false pos found for prediction {}'.format(i))
            # false_pos += 1

    # for (x,y,w,h) in faces:
    #     print(x,y,w,h)
    #     cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)

    print('labels:')
    print(face_labels[i])


    plt.figure()
    plt.imshow(img)
    plt.show()



# The main method is used to compare the accuracies of the FaceNet detector and Haar Cascade detector
# 
def test_accuracy():
    total_correct, total_faces, total_false_pos = 0, 0, 0
    start_time = datetime.now()
    for fold_num in [2,3,4,5]:
        img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
        with open(img_list_file, 'r') as f:
            file_names = [x.rstrip() for x in f.readlines()]

        face_images = get_images_from_file_list(file_names)
        face_labels = retrieve_face_list(fold_num)

        with open(img_list_file, 'r') as f:
            file_names = [x.rstrip() for x in f.readlines()]


        # num_correct, num_faces, false_pos = test_detection(fold_num, file_names, face_images, face_labels)
        num_correct, num_faces, false_pos = test_cnn_detection(fold_num, file_names, face_images, face_labels)

        total_correct += num_correct
        total_faces += num_faces
        total_false_pos += false_pos

    delta = datetime.now() - start_time
    print('******** TOTALS ***********')
    print('found {}/{} faces'.format(total_correct, total_faces))
    print('total false pos: {}'.format(total_false_pos))
    print('accuracy: {}'.format(total_correct/total_faces))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(delta))

def test_one_image():
    fold_num = 5
    img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
    with open(img_list_file, 'r') as f:
        file_names = [x.rstrip() for x in f.readlines()]

    face_images = get_images_from_file_list(file_names)
    face_labels = retrieve_face_list(fold_num)
    test_on_one_image(file_names, face_labels)

def test_on_manual_labels():
    img_list_file = 'img/manual/image_list.txt'
    with open(img_list_file, 'r') as f:
        file_names = [x.rstrip() for x in f.readlines()]
    face_images = get_images_from_file_list(file_names)

    start_time = datetime.now()
    face_labels = retrieve_manual_face_labels(1, file_names)

    # num_correct, num_faces, false_pos = test_detection(1, file_names, face_images, face_labels)
    num_correct, num_faces, false_pos = test_cnn_detection(1, file_names, face_images, face_labels)

    
    delta = datetime.now() - start_time
    print('found {}/{} faces'.format(num_correct, num_faces))
    print('total false pos: {}'.format(false_pos))
    print('accuracy: {}'.format(num_correct/num_faces))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(delta))
    

def test_haar():
    folds = [2,3,4,5]
    # prepare fold info
    fold_to_info_dict = {}
    for fold_num in folds:
        img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
        with open(img_list_file, 'r') as f:
            file_names = [x.rstrip() for x in f.readlines()]
        face_images = get_images_from_file_list(file_names)
        face_labels = retrieve_face_list(fold_num)
        fold_to_info_dict[fold_num] = (file_names, face_images, face_labels)


    for min_neighbors in [0,1,2,3,4,5]:
        scale = 1.05
        while scale < 1.5:
            start = datetime.now()
            total_correct, total_faces, total_false_pos = 0, 0, 0
            for fold_num in folds:
                file_names, face_images, face_labels = fold_to_info_dict[fold_num]
                num_correct, num_faces, false_pos = test_haar_detection(fold_num, file_names, face_images, face_labels, scale, min_neighbors)
                
                total_correct += num_correct
                total_faces += num_faces
                total_false_pos += false_pos

            delta = datetime.now() - start
            print('minNeighbors={}, scale={}: accuracy={}, avgFalsePos={}, ttlFP={}, timing={}'.format(min_neighbors, scale, total_correct/total_faces, total_false_pos/len(folds), total_false_pos, delta))
            scale += .05

def test_dlib():
    folds = [2,3,4,5]
    # prepare fold info
    fold_to_info_dict = {}
    for fold_num in folds:
        img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
        with open(img_list_file, 'r') as f:
            file_names = [x.rstrip() for x in f.readlines()]
        face_images = get_images_from_file_list(file_names)
        face_labels = retrieve_face_list(fold_num)
        fold_to_info_dict[fold_num] = (file_names, face_images, face_labels)

    for upscale in [0,1,2,3]:
        start = datetime.now()
        total_correct, total_faces, total_false_pos = 0, 0, 0
        for fold_num in folds:
            file_names, face_images, face_labels = fold_to_info_dict[fold_num]
            num_correct, num_faces, false_pos = test_dlib_detection(fold_num, file_names, face_images, face_labels, upscale)
            
            total_correct += num_correct
            total_faces += num_faces
            total_false_pos += false_pos

        delta = datetime.now() - start
        print('upscale={}: accuracy={}, avgFalsePos={}, ttlFP={}, time: {}'.format(upscale, total_correct/total_faces, total_false_pos/len(folds), total_false_pos, delta))

if __name__ == "__main__":
    # main()
    test_haar()
    # test_dlib()
    # test_one_image()
    # test_on_manual_labels()
