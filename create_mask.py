"""
Создание масок изображений для unet.

Имеем 13 директорий датасета. В каждом из них есть папки с 
аннотицями и изображениями. Для каждого изображения строим 
маску. Далее матрицы маски и изображения сохраняем в json файл
"""

import get_mask
import cv2
import os
import utils


# пути аннотаций, изображений и папки
# для будущего расположения масок и изображений
ann_dirs = [f"dataset\\ds{i}\\ann" for i in range(1, 14)]
orig_img_dirs = [f'dataset\\ds{i}\\img' for i in range(1, 14)]
out_path = 'sply_json'

def main(ann_dirs, orig_img_dirs, out_path):

    for i in range(len(orig_img_dirs)):
        images = os.listdir(orig_img_dirs[i])
        for image_name in images:
            # получаем изображение
            img = utils.get_img(os.path.join(orig_img_dirs[i], image_name)) 
            
            # получаем маску
            ann_path = get_mask.get_ann_path(ann_dirs[i], image_name)
            ann = get_mask.GetMask(ann_path)
            mask = ann.create_mask()
           
            # сохраняем в json файл
            out_json_name = os.path.join(out_path, image_name.split('.')[0])
            utils.create_json(img, mask, out_json_name)
            print(f"Маска для изображения {image_name} размером {img.shape} готова и сохранена в \n{out_json_name}", )

if __name__ == "__main__":
    main(ann_dirs, orig_img_dirs, out_path)
