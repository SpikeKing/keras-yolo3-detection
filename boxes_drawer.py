#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/11
"""
import cv2
import os

from root_dir import ROOT_DIR


def draw_img(image, boxes):
    img = cv2.imread(image)

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        i1_pt1 = (int(x_min), int(y_min))
        i1_pt2 = (int(x_max), int(y_max))
        cv2.rectangle(img, pt1=i1_pt1, pt2=i1_pt2, thickness=3, color=(255, 0, 255))

    cv2.imshow('Image', img)
    cv2.imwrite('./data/img_346.bbox.jpg', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = os.path.join(ROOT_DIR, 'dataset', 'originalPics', '2002/08/22/big/img_734.jpg')
    print(img_path)
    boxes = [(87, 0, 229, 215), (149, 263, 163, 283), (178, 260, 192, 282), (205, 270, 220, 291)]
    draw_img(img_path, boxes)
