import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
import imutils
from scipy import misc
from PIL import Image
from detect_face import dlib_face_detect, haar_face_detect, cnn_face_detect
import math
import pickle
import dlib
import cv2
import os
from helper import *

GLASSES_WIDTH_PX = 600
GLASSES_HEIGHT_PX = 209
MANUAL_FACE_LABELS_FILE = 'img/manual/face_labels.pkl'
MANUAL_EYE_LABELS_FILE = 'img/manual/eye_labels.pkl'

eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_eye.xml')

dlib_face_struct_predictor = dlib.shape_predictor('src/dlib/shape_predictor_68_face_landmarks.dat')

def get_bounding_box(list_of_coords):
    x_min = min([x[0] for x in list_of_coords])
    x_max = max([x[0] for x in list_of_coords])
    y_min = min([y[1] for y in list_of_coords])
    y_max = max([y[1] for y in list_of_coords])

    return (x_min, y_min, x_max-x_min, y_max-y_min)

def hog_detect_eyes(image, face_list, dlib_rects=True, bounding_boxes=True):
    # if not a list of dlib rectangles, convert faces so that they are
    if not dlib_rects:
        new_face_list = []
        for face in face_list:
            if face is None or len(face) == 0:
                new_face_list.append(None)  
                continue
            (x,y,w,h) = round_int(face)
            new_face_list.append(dlib.rectangle(x, y, x+w, y+h))
        face_list = new_face_list

    # Iterates over set of faces in each image
    face_list = sorted(face_list, key=lambda x: (x.left(),x.top()))
    eye_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for rect in face_list:
        if rect is None:
            eye_list.append([])
            continue        
        shape = dlib_face_struct_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # clone = image.copy()
        # for x,y in shape:
        #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        # cv2.imshow('img',clone)
        # cv2.waitKey(0)

        a, b = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        c, d = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']

        # print(np.array(shape[a:b]))

        # sort your eyes
        right = np.array(shape[a:b])
        right = sorted(right, key=lambda point: (point[0],point[1]))
        left = np.array(shape[c:d])
        left = sorted(left, key=lambda point: (point[0],point[1]))

        eye_tuple = (right, left)
        if bounding_boxes:
            eye_tuple = (get_bounding_box(eye_tuple[0]), get_bounding_box(eye_tuple[1]))

        eye_list.append(eye_tuple)
        # print(eye_set)
    if bounding_boxes:
        eye_list = sorted(eye_list, key=lambda x: (x[0][0], x[0][1]))    
    return eye_list

def haar_detect_eyes(image, face_list, dlib_rects=True):
    # if list of dlib rectangles, convert faces to (x,y,w,h)
    if dlib_rects:
        new_face_list = []
        for rect in face_list:
            new_face_list.append((rect.left(),rect.top(),rect.right()-rect.left(),rect.bottom()-rect.top()))
        face_list = new_face_list

    # face_list = sorted(face_list, key=lambda face: (face[0],face[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_list = []
    for face in face_list:
        if face is None or len(face) == 0:
            eye_list.append([])
            continue

        x,y,w,h = round_int(face)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # sorts so that right_eye is always first
        eyes = sorted(eyes, key=lambda eye: (eye[0], eye[1]))
        
        # shift by dimensions of face to realign in orig image
        for eye in eyes:
            eye[0] += x
            eye[1] += y

        eye_list.append(eyes)

    return eye_list

def display_eyes(image, eye_list):
    im = image.copy()
    for eye_pair in eye_list:
        for x,y,w,h in eye_pair:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)

    cv2.imshow('img',im)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


# given a file name and a list of eyes in the file, place glasses
# over each pair of eyes. The result displays the original file
# with glasses over each pair of eyes that was detected
def place_glasses(file_name, eye_list):
    # pre-opens the files
    face_im = Image.open(file_name)
    glasses_im = Image.open('img/black-sunglasses.png')
    # face_im2 = cv2.imread(file_name)

    # iterate over each pair of eyes in the given image
    for right_eye, left_eye in eye_list:
        # compute the centroids of eyes

        # for (x,y) in right_eye:
        #     cv2.circle(face_im2, (x, y), 1, (0, 0, 255), -1)

        length = len(right_eye)
        print(right_eye)
        sum_x = np.sum([x[0] for x in right_eye])
        sum_y = np.sum([x[1] for x in right_eye])
        right_cx, right_cy = int(round(sum_x/length)), int(round(sum_y/length))
        
        # for (x,y) in left_eye:
        #     cv2.circle(face_im2, (x, y), 1, (0, 0, 255), -1)
        
        length = len(left_eye)
        sum_x = np.sum([x[0] for x in left_eye])
        sum_y = np.sum([x[1] for x in left_eye])
        left_cx, left_cy = int(round(sum_x/length)), int(round(sum_y/length))

        # cv2.line(face_im2, (right_cx,right_cy), (left_cx,left_cy), (0,255,0))

        # calculates midpoint between eyes, angle at which eyes are sloped,
        # and distance between eyes
        mid_x, mid_y = (right_cx + left_cx)/2, (right_cy + left_cy)/2
        angle = math.atan((left_cy - right_cy)/(left_cx - right_cx))

        degrees = angle * 180 / math.pi
        diff_x, diff_y = left_cx - right_cx, left_cy - right_cy
        distance = math.sqrt(diff_x*diff_x + diff_y*diff_y)

        # cv2.line(face_im2, (right_cx, right_cy), (int(right_cx+math.cos(angle)*distance), right_cy), (255,0,0))
        # cv2.imshow('img',face_im2)
        # cv2.waitKey(0)
        # continue


        glasses = glasses_im.copy()

        # resize the glasses to appropriate scale
        scale = distance*2/GLASSES_WIDTH_PX        
        glasses = glasses.resize((int(round(GLASSES_WIDTH_PX*scale)), int(round(GLASSES_HEIGHT_PX*scale))), Image.ANTIALIAS)
        
        # rotate the glasses
        glasses = glasses.rotate(-1*degrees, expand=True)
        
        # compute the size of the new image, and find upper left coordinate for placement on image
        im_width, im_height = glasses.size
        pos_x, pos_y = int(round(mid_x - im_width/2)), int(round(mid_y - im_height/2))

        face_im.paste(glasses, (pos_x, pos_y), glasses)
    
    face_im.show()

# def sort_eyes(list_of_eyes):
#     # hog eye outline of points
#     # if len(eye_set[0]) > 4:
#     for i, eye_set in enumerate(list_of_eyes):

#         list_of_eyes[i] = sorted(key=lambda eye_set: (eye_set[0][0][0], eye_set[0][0][1]))
#     for eye_set in list_of_eyes:
#         for eye_points in eye_set:
#             eye_points.sort(key=lambda point: (point[0], point[1]))
#         eye_set.sort(key=lambda points: (points[0][0], points[0][1]))
#     list_of_eyes.sort(key=lambda eye_set: (eye_set[0][0][0], eye_set[0][0][1]))
#     return list_of_eyes
#     # set of eyes
#     # else:

# Computs the distance between the centers of two eyes
def eye_dist(eye1, eye2):
    x1,y1,w1,h1 = eye1
    x2,y2,w2,h2 = eye2
    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2
    dx, dy = cx1 - cx2, cy1 - cy2
    return math.sqrt(dx*dx + dy*dy)

def eyes_equal(eye1, eye2):
    if eye1 is None and eye2 is None:
        return True
    if eye1 is None or eye2 is None:
        return False

    return np.all([a == b for a,b in zip(eye1, eye2)])

# Finds the closest eye in the list of eye detections for the
# given eye in eye label (right and left eye)
def find_closest_eyes(eye_pair, eye_list):
    right_closest, left_closest = None, None
    right_dist, left_dist = float('inf'), float('inf')
    right, left = eye_pair
    for eye in eye_list:
        if right_closest is None or eye_dist(right, eye) < right_dist:
            right_closest, right_dist = eye, eye_dist(right, eye)

        if left_closest is None or eye_dist(left, eye) < left_dist:
            left_closest, left_dist = eye, eye_dist(left, eye)

    # if both have same eye, we have a problem
    if eyes_equal(right_closest, left_closest):
        # make sure there were less than two eyes to choose from
        if len(eye_list) >= 2:
            print('why am i here?')
            print(right)
            print(left)
            print(right_closest)
            print(left_closest)
            print('and now the eyes')
            print(eye_list)
        assert len(eye_list) < 2

        if right_dist < left_dist:
            left_closest = None
        else:
            right_closest = None

    return (right_closest, left_closest)

def meets_discrete_thresh(label, predict):
    if predict is None:
        return 0
    if continuous_score(label, predict) > .3:
        return 1

    x_l, y_l, w_l, h_l = label
    x_p, y_p, w_p, h_p = predict
    center_lx, center_ly = x_l + w_l/2, y_l + h_l/2
    center_px, center_py = x_p + w_p/2, y_p + h_p/2

    if (abs(center_lx - center_px) < .5*w_l and abs(center_ly - center_py) < .5*h_l):
        return 1
    return 0
# Computes accuracy, continuous score, and false positives for a given list of labels
# Handles for a fold of images (meaning in the label list, at each index i there is a 
# list of eye pairs for each face in image i).
# Returns the accuracy, continous score, number of false positives of fold
def compute_scores(label_list, eye_predictions):
    total_eyes, num_correct, cont_score, false_pos = 0, 0, 0, 0
    
    # Iterate over each image
    for eye_list, image_predictions in zip(label_list, eye_predictions):
        # iterate over a pair of eyes and eye predictions for each face in the image
        for eye_pair, predictions in zip(eye_list, image_predictions):
            right, left = eye_pair
            r_predict, l_predict = find_closest_eyes(eye_pair, predictions)

            # operations for discrete score
            total_eyes += 2
            correct = 0
            # print('CHECKING')
            # print(right)
            # print(r_predict)
            correct += meets_discrete_thresh(right, r_predict)
            correct += meets_discrete_thresh(left, l_predict)

            # for continuous score
            cont_score += continuous_score(right, r_predict)
            cont_score += continuous_score(left, l_predict)

            # the number of false positives is the number of predicted eyes that
            # were not correctly matched
            false_pos += len(predictions) - correct
            num_correct += correct

    return num_correct/total_eyes, cont_score, false_pos


def main():
    #######################
    # Prepare files
    # with open('img/manual/image_list.txt', 'r') as f:
        # file_names = [x.rstrip() for x in f.readlines()]

    fold_num = 2
    img_list_file = 'img/FDDB-folds/FDDB-fold-{:02}.txt'.format(fold_num)
    with open(img_list_file, 'r') as f:
        file_names = [x.rstrip() for x in f.readlines()]

    image_list = get_images_from_file_list(file_names)
    
    with open(MANUAL_FACE_LABELS_FILE, 'rb') as f:
        face_labels = pickle.load(f)
    with open(MANUAL_EYE_LABELS_FILE, 'rb') as f:
        eye_labels = pickle.load(f)

    

    ###########################

    # CHANGE HERE HOW TO GET FACES:
    # face_list = face_labels
    face_list = []
    count = 0
    for image in image_list:
        count += 1
        # face_list.append(haar_face_detect(image, 1.25, 5))
        # faces = cnn_face_detect(image)
        # faces = sorted(faces, key=lambda x: (x[0],x[1]))
        
        # if count == 2:
        #     print('ive hit this')
        #     faces = [faces[0], [], faces[1]]
        # print(faces)
        # face_list.append(faces)
        face_list.append(dlib_face_detect(image))



    eye_list = []
    # for file, image, faces, eyes2 in zip(file_names, image_list, face_list, eye_labels):
    for file, image, faces in zip(file_names, image_list, face_list):
        # Change detection method + how the faces are given
        # eyes contains a list of detected eyes for each face in the image
        # eyes = haar_detect_eyes(image, faces, dlib_rects=False)
        eyes = hog_detect_eyes(image, faces, dlib_rects=True, bounding_boxes=False)
        
        
        eye_list.append(eyes)

    # accuracy, cont_score, false_pos = compute_scores(eye_labels, eye_list)
    # print('Accuracy: {}'.format(accuracy))
    # print('continuous score: {}'.format(cont_score))
    # print('num false pos: {}'.format(false_pos))

    count = 0
    for file, eyes in zip(file_names, eye_list):
        count += 1
        if count > 15:
            break
        file2 = 'img/FDDB-pics/{}.jpg'.format(file)
        place_glasses(file2, eyes)

    # for i in range(20):
        # display_eyes(image_list[i], eye_list[i])

     

    

def main2():
    img_name = 'img/olga.jpg'
    im = cv2.imread(img_name)
    faces = dlib_face_detect(im)
    eyes = hog_detect_eyes(im, faces)
    place_glasses(img_name, eyes)



if __name__ == "__main__":
    main()
    # main2()
    # rect1 = (0, 4, )
    # rect2 = (6, 0, 5, 8)

