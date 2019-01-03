#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""

"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    def __init__(self):
        self.anchors_path = 'configs/yolo_anchors.txt'  # Anchors
        self.model_path = 'model_data/yolo_weights.h5'  # 模型文件
        self.classes_path = 'configs/coco_classes_ch.txt'  # 类别文件

        # self.model_path = 'model_data/ep074-loss26.535-val_loss27.370.h5'  # 模型文件
        # self.classes_path = 'configs/wider_classes.txt'  # 类别文件

        self.score = 0.60
        self.iou = 0.45
        # self.iou = 0.01
        self.class_names = self._get_class()  # 获取类别
        self.anchors = self._get_anchors()  # 获取anchor
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw

        self.colors = self.__get_colors(self.class_names)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def __get_colors(names):
        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(names), 1., 1.)
                      for x in range(len(names))]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)  # 转换~
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(self.anchors)  # anchors的数量
        num_classes = len(self.class_names)  # 类别数

        self.yolo_model = yolo_body(Input(shape=(416, 416, 3)), 3, num_classes)
        self.yolo_model.load_weights(model_path)  # 加载模型参数

        print('{} model, {} anchors, and {} classes loaded.'.format(model_path, num_anchors, num_classes))

        # 根据检测参数，过滤框
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output, self.anchors, len(self.class_names),
            self.input_image_shape, score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()  # 起始时间

        if self.model_image_size != (None, None):  # 416x416, 416=32*13，必须为32的倍数，最小尺度是除以32
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print('detector size {}'.format(image_data.shape))
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 检测出的框

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 厚度
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  # 类别
            box = out_boxes[i]  # 框
            score = out_scores[i]  # 执行度

            label = '{} {:.2f}'.format(predicted_class, score)  # 标签
            draw = ImageDraw.Draw(image)  # 画图
            label_size = draw.textsize(label, font)  # 标签文字

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))  # 边框

            if top - label_size[1] >= 0:  # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):  # 画框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(  # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
            del draw

        end = timer()
        print(end - start)  # 检测执行时间
        return image

    def detect_objects_of_image(self, img_path):
        image = Image.open(img_path)
        assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维
        # print('detector size {}'.format(image_data.shape))

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('out_boxes: {}'.format(out_boxes))
        # print('out_scores: {}'.format(out_scores))
        # print('out_classes: {}'.format(out_classes))

        img_size = image.size[0] * image.size[1]
        objects_line = self._filter_boxes(out_boxes, out_scores, out_classes, img_size)
        return objects_line

    def _filter_boxes(self, boxes, scores, classes, img_size):
        res_items = []
        for box, score, clazz in zip(boxes, scores, classes):
            top, left, bottom, right = box
            box_size = (bottom - top) * (right - left)
            rate = float(box_size) / float(img_size)
            clz_name = self.class_names[clazz]
            if rate > 0.05:
                res_items.append('{}-{:0.2f}'.format(clz_name, rate))
        res_line = ','.join(res_items)
        return res_line

    def close_session(self):
        self.sess.close()


def detect_img_for_test():
    yolo = YOLO()
    img_path = './dataset/vDaPl5QHdoqb2wOaVql4FoJWNGglYk.jpg'
    image = Image.open(img_path)
    r_image = yolo.detect_image(image)
    yolo.close_session()
    r_image.save('xxx.png')


def test_of_detect_objects_of_image():
    yolo = YOLO()
    img_path = './dataset/vDaPl5QHdoqb2wOaVql4FoJWNGglYk.jpg'
    objects_line = yolo.detect_objects_of_image(img_path)
    print(objects_line)


if __name__ == '__main__':
    # detect_img_for_test()
    test_of_detect_objects_of_image()
