import json
import numpy as np
import cv2


def get_img(img_path):
    return cv2.imread(img_path)


def create_json(img, mask, image_name):
    outdata = {'image': img.tolist(), 'mask': mask.tolist()}
    with open(f'{image_name}.json', 'w') as outfile:
        json.dump(outdata, outfile)


def transform_img(img):
    img = img.astype('float32')
    return img


def save_img_and_mask(outdir, image_name, img, mask):

    # сохранение изображения в директории new_data\images
    cv2.imwrite(outdir + '\\images\\' + image_name, img)

    # сохранение маски в директории new_data\masks
    cv2.imwrite(outdir + '\\masks\\' + image_name, mask)