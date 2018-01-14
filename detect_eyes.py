import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
import imutils
from scipy import misc
from PIL import Image
from detect_face import hog_face_detect
import math
import pickle
import dlib
import cv2
import os

GLASSES_WIDTH_PX = 600
GLASSES_HEIGHT_PX = 209

eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_eye.xml')

dlib_face_struct_predictor = dlib.shape_predictor('src/dlib/shape_predictor_68_face_landmarks.dat')

def dlib_detect_eyes(image, face_list, dlib_rects=True):
    # if not a list of dlib rectangles, convert faces so that they are
    if not dlib_rects:
        new_face_list = []
        for (x,y,w,h) in face_list:    
            face_rects.append(dlib.rectangle(x, y, x+w, y+h))
        face_list = new_face_list

    # Iterates over set of faces in each image
    eye_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for rect in face_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        shape = dlib_face_struct_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        a, b = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        c, d = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        # print(np.array(shape[a:b]))
        eye_tuple = (np.array(shape[a:b]), np.array(shape[c:d]))
        eye_list.append(eye_tuple)
        # print(eye_set)
    
    return eye_list

def place_glasses(file_name, eye_list):
    # pre-opens the files
    face_im = Image.open(file_name)
    glasses_im = Image.open('img/black-sunglasses.png')

    # iterate over each pair of eyes in the given image
    for right_eye, left_eye in eye_list:
        # compute the centroids of eyes
        length = len(right_eye)
        sum_x = np.sum(right_eye[:,0])
        sum_y = np.sum(right_eye[:,1])
        right_cx, right_cy = int(round(sum_x/length)), int(round(sum_y/length))

        length = len(left_eye)
        sum_x = np.sum(left_eye[:,0])
        sum_y = np.sum(left_eye[:,1])
        left_cx, left_cy = int(round(sum_x/length)), int(round(sum_y/length))

        # calculates midpoint between eyes, angle at which eyes are sloped,
        # and distance between eyes
        mid_x, mid_y = (right_cx + left_cx)/2, (right_cy + left_cy)/2
        angle = math.atan((left_cy - right_cy)/(left_cx - right_cx))
        degrees = angle * 180 / math.pi
        diff_x, diff_y = left_cx - right_cx, left_cy - right_cy
        distance = math.sqrt(diff_x*diff_x + diff_y*diff_y)

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




def main():
     
    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    # img_glasses = img_glasses[:,:,0:3]
    # orig_glasses_height, orig_glasses_width = img_glasses.shape[:2]

    ## delet tis
    img_list_file = 'img/FDDB-folds/FDDB-fold-02.txt'
    with open(img_list_file, 'r') as f:
        file_list = [x.rstrip() for x in f.readlines()]

    image_list = [cv2.imread('img/FDDB-pics/{}.jpg'.format(file)) for file in file_list]

    ## and tis
    # rectangle_file = 'img/FDDB-folds/FDDB-fold-02-rectangleList.pkl'
    # with open(rectangle_file, 'rb') as f:
    #     face_list = pickle.load(f)



    index_num = 5
    # img = cv2.imread('img/FDDB-pics/{}.jpg'.format(file_list[index_num]))
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_list[index_num]

    images = [image_list[index_num]]
    face_list = hog_face_detect(images)
    eye_list = dlib_detect_eyes(images, face_list)

    # iterate over each image and set of eyes
    
    for img, eye_set in zip(images, eye_list):
        clone = img.copy()
        # iterate over eye pairs for each face in img
        for right_eye, left_eye in eye_set:
            # print(right_eye)
            # iterate over each of the points in right and left eyes
            for (x,y) in right_eye:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            for (x,y) in left_eye:
                # print(x,y)
                cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)

            # compute centroids of eyes
            length = len(right_eye)
            sum_x = np.sum(right_eye[:,0])
            sum_y = np.sum(right_eye[:,1])
            right_cx, right_cy = int(round(sum_x/length)), int(round(sum_y/length))

            length = len(left_eye)
            sum_x = np.sum(left_eye[:,0])
            sum_y = np.sum(left_eye[:,1])
            left_cx, left_cy = int(round(sum_x/length)), int(round(sum_y/length))
            # print(right_cx)
            # cv2.circle(clone, (right_cx, right_cy), 1, (0, 255, 0), -1)
            # cv2.circle(clone, (left_cx, left_cy), 1, (0, 255, 0), -1)

            # now apply a pair of glasses
            mid_x, mid_y = (right_cx + left_cx)/2, (right_cy + left_cy)/2
            angle = math.atan((left_cy - right_cy)/(left_cx - right_cx))
            degrees = angle * 180 / math.pi
            diff_x, diff_y = left_cx - right_cx, left_cy - right_cy
            distance = math.sqrt(diff_x*diff_x + diff_y*diff_y)

            file = file_list[index_num]
            face_im = Image.open('img/FDDB-pics/{}.jpg'.format(file))
            glasses_im = Image.open('img/black-sunglasses.png')

            scale = distance*2/600
            glasses_im = glasses_im.resize((int(round(600*scale)), int(round(209*scale))), Image.ANTIALIAS)
            glasses_im = glasses_im.rotate(-1*degrees, expand=True)
            im_width, im_height = glasses_im.size
            pos_x, pos_y = int(round(mid_x - im_width/2)), int(round(mid_y - im_height/2))

            face_im.paste(glasses_im, (pos_x, pos_y), glasses_im)
            face_im.show()






            

            # print(sum_x)
        # cv2.imshow("Image", clone)
        # cv2.waitKey(0)
    


    # for tup in faces:
    #     (x, y, w, h) = (int(round(a)) for a in tup)
    #     cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)

    #     roi_gray = gray_img[y:y+h, x:x+w]
    #     roi_color = img[y:y+h, x:x+w]

    #     shape = predictor(gray, rect)
    #     shape = face_utils.shape_to_np(shape)
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def rotate_glasses():
    IMG_PIXEL_WIDTH = 600
    IMG_PIXEL_HEIGHT = 209

    img_glasses = cv2.imread('img/black-sunglasses.png',-1)
    # rotated = imutils.rotate(img_glasses, 45)
    # rotated = imutils.rotate_bound(img_glasses, -45)
    # cv2.imshow("Rotated (Problematic)", rotated)
    # cv2.waitKey(0)
    face = Image.open('img/FDDB-pics/2002/08/16/big/img_166.jpg')
    im = Image.open('img/black-sunglasses.png')
    # im.thumbnail((600,100), Image.ANTIALIAS)
    im = im.resize((600,300), Image.ANTIALIAS)
    im = im.rotate(45, expand=True)
    # im.show()

    # center of glasses at width/2, height/2
    print(im.size)
    # print(size(im))


    face.paste(im, (0, 0), im)
    face.paste(im, (100,100), im)
    face.show()

    # original pixel size of image is 600x209
    # im.thumbnail((600,100), Image.ANTIALIAS)
    # im.rotate(45).show()
    # im.rotate(45, expand=True).show()

def main2():
    im = cv2.imread('img/olga.jpg')
    faces = hog_face_detect(im)
    eyes = dlib_detect_eyes(im, faces)
    place_glasses('img/olga.jpg', eyes)



if __name__ == "__main__":
    # main()
    main2()
    # rotate_glasses()

