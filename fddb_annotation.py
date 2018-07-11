#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/10
"""
import os

from project_utils import *
from root_dir import DATASET, ROOT_DIR


def generate_train_file(bbx_folder, data_folder, out_file):
    bbx_list, bbx_name = traverse_dir_files(bbx_folder)
    bbx_file_list = []
    for bbx_item in bbx_list:
        if 'ellipseList' in bbx_item:
            bbx_file_list.append(bbx_item)
    print('物体框列表: %s' % bbx_file_list)
    data_list, _ = traverse_dir_files(data_folder)

    for bbx_file in bbx_file_list:
        print('文件名: %s' % bbx_file)
        data_lines = read_file(bbx_file)

        file_path = None
        n_face = 0
        box_line_list = []
        for count, data_line in enumerate(data_lines):
            if n_face == 0 and not file_path:
                file_path = data_line
                file_path = os.path.join(ROOT_DIR, 'dataset', 'originalPics', file_path + '.jpg')
                continue

            if n_face == 0 and file_path:
                n_face = int(data_line)
                continue

            if n_face != 0:
                n_face -= 1
                value_list = data_line.split(' ')
                x_min = int(float(value_list[3]) - float(value_list[1]))
                x_max = int(float(value_list[3]) + float(value_list[1]))
                y_min = int(float(value_list[4]) - float(value_list[0]))
                y_max = int(float(value_list[4]) + float(value_list[0]))
                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                box_line = ','.join([str(x_min), str(y_min), str(x_max), str(y_max), str(0)])
                box_line_list.append(box_line)
                if n_face == 0:
                    if box_line_list:
                        img_line = ' '.join([file_path] + box_line_list)
                        write_line(out_file, img_line)
                        box_line_list = []
                    file_path = None
                continue

    res_lines = read_file(out_file)
    print('数据行: %s' % len(res_lines))


if __name__ == '__main__':
    bbx_folder = os.path.join(DATASET, 'FDDB-folds')
    data_folder = os.path.join(DATASET, 'originalPics')
    out_file = os.path.join(DATASET, 'FDDB_train.txt')
    generate_train_file(bbx_folder, data_folder, out_file)
