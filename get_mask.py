"""
функции и классы для создания маски
"""

import json
import cv2
import numpy as np
import os
import zlib
import base64

class AnnData():
    """
    класс содержит размеры изображения и его объекты
    
    """

    def __init__(self, ann_path):
        self.ann_path = ann_path
        self.ann_data = self._get_anndata()
        self.img_size = self.ann_data['size'] # keys: height, widht
        self.objects = self.ann_data['objects']

    def _get_anndata(self):
        """Функция возвращает данные из аннотации"""       
        with open(self.ann_path) as f:
            ann_data = json.load(f)
        return ann_data


class PersonPoly:
    """
    Маска класса person_poly

    """

    def __init__(self, obj):
        self.exterior = obj['points']['exterior']


class PersonBmp:
    """
    Маска класса person_bmp

    """

    def __init__(self, obj):
        self.coded_mask = obj["bitmap"]["data"]
        self.mask_coordinate = obj["bitmap"]["origin"]
        

class GetMask(AnnData):
    """
    Создает маску изображения
    """

    def __init__(self, ann_path):
        super().__init__(ann_path)

    def create_mask(self):
        mask_objects = self._get_objects()
        mask = self._create_matrix(mask_objects)
        return mask

    def _get_objects(self):
        mask_objs = {'pp': [], 'pb': []}
        for obj in self.objects:
            if obj['classTitle'] == 'person_poly':
                mask_objs['pp'].append(PersonPoly(obj))
            elif obj['classTitle'] == 'person_bmp':
                mask_objs['pb'].append(PersonBmp(obj))
        return mask_objs

    def _create_matrix(self, mask_objects):
        obj_pp = mask_objects['pp']
        obj_pb = mask_objects['pb']
        if len(obj_pp) != 0 and len(obj_pb) == 0:
            mask = self._create_person_poly_mask(obj_pp)
        elif len(obj_pp) == 0 and len(obj_pb) != 0:
            mask = self._create_person_bmp_mask(obj_pb)
        elif len(obj_pp) != 0 and len(obj_pb) != 0:
            mask = self._create_person_polybmp_mask(obj_pp, obj_pb)
        return mask

    def _create_person_poly_mask(self, obj_pp):
        """Функция создает маску person poly"""
        
        exterior = np.asarray(obj_pp[0].exterior)
        exterior = exterior.reshape((-1,1,2))
        #размеры картинки
        height, width = self.img_size['height'], self.img_size['width']
        # строим маску
        blank_image = np.zeros((height,width), np.uint8)
        mask = cv2.fillPoly(blank_image, [exterior], 255)
        return mask

    def _create_person_bmp_mask(self, obj_pb):
        """Функция создает маску person bmp"""
        
        im_masks = []
        for obj in obj_pb:
            matrix_mask = self._get_matrix_mask(obj.coded_mask)
            im_mask = self._create_im_mask(matrix_mask, obj.mask_coordinate)
            im_masks.append(im_mask)

        mask = np.zeros((self.img_size['height'], self.img_size['width']), np.uint8)
        mask = self._matrices_to_mask(im_masks, mask)
        return mask

    def _create_person_polybmp_mask(self, obj_pp, obj_pb):
        """Функция создает маску из классов person_poly и person_bmp"""

        mask_pp = self._create_person_poly_mask(obj_pp)
        mask_pb = self._create_person_bmp_mask(obj_pb)
        for i, string in enumerate(mask_pp):
            for j, pix in enumerate(string):
                if pix == 255:
                    mask_pb[i][j] = 255
        return mask_pb            

    def _get_matrix_mask(self, coded_mask):
        decoded_str = zlib.decompress(base64.b64decode(coded_mask))
        array = np.frombuffer(decoded_str, np.uint8)
        matrix_mask = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)[:, :, 1].astype(bool)
        return matrix_mask
    
    def _create_im_mask(self, matrix_mask, mask_coord):
        im_mask = np.zeros((self.img_size['height'], self.img_size['width']), dtype=np.uint8)

        for i, row in enumerate(matrix_mask):
            for j, pix in enumerate(row):
                if pix:
                    im_mask[i + mask_coord[1]][j + mask_coord[0]] = 255
        
        return im_mask

    def _matrices_to_mask(self, matrices, mask):
        for index, matrix in enumerate(matrices):
            for i, string in enumerate(matrix):
                for j, pix in enumerate(string):
                    if pix == 255:
                        mask[i][j] = 255
        return mask

    def get_img(self, img_path='', img_name=''):
        """Возращает изображение"""
        img_path = os.path.join(img_path, img_name)
        img = cv2.imread(img_path) 
        return img


def get_ann_path(ann_dir='',img_name=''):
    """
    Функция возвращает путь аннотации картинки,
    которую подаем на вход
    """
    ann_path = os.path.join(ann_dir, f'{img_name}.json')
    return ann_path
