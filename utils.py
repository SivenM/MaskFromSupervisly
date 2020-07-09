import json
import numpy as np
import cv2


def get_img(img_path):
    return cv2.imread(img_path)


def create_json(img, mask, image_name):
    outdata = {'image': img, 'mask': mask}
    with open('data.json', 'w') as outfile:
        json.dump(outdata, image_name)


def transform_img(img):
    img = img.astype('float32')
    return img

