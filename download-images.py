import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513'   
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1
    
    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  

def store_pos_images():
    pos_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09618957'
    pos_image_urls = urllib.request.urlopen(pos_images_link).read().decode()
    pic_num = 1

    if not os.path.exists('img/test'):
        os.makedirs('img/test')

    for num, i in enumerate(pos_image_urls.split('\n')):
        if num < 600:
            continue
        if pic_num > 400:
            break
        try:
            print("{}, {}".format(num,i))
            urllib.request.urlretrieve(i, "img/test/pos-{}.jpg".format(pic_num))
            # img = cv2.imread("pos/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            # resized_image = cv2.resize(img, (100, 100))
            # cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
        except Exception as e:
            print(str(e))

def resize_images():
    num_images = len(os.listdir('img/pos'))
    for i in range(1, num_images+1):
        pass

def rename_images():
    files = os.listdir('img/pos')
    count = 1
    for file in files:
        if ".jpg" not in file:
            continue
        # print(str(i) + " " + file)
        os.rename('img/pos/{}'.format(file), 'img/pos/pos-{}.jpg'.format(count))
        count += 1

# store_pos_images()
# rename_images()
# print(len(os.listdir('img/pos')))

# print(len(os.listdir('img')))
# print(len([name for name in os.listdir('img') if os.path.isfile(name)]))