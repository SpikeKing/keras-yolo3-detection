#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/1/3
"""

from project_utils import *
from root_dir import ROOT_DIR
from yolo3_predict import YOLO


def generate_data(file_name):
    print('执行开始')

    out_file = os.path.join(ROOT_DIR, 'dataset', file_name)
    create_file(out_file)

    up_folder = os.path.abspath(os.path.join(ROOT_DIR, '..'))
    data_folder = os.path.join(up_folder, 'data_set', 'XX-Images-57w-416')

    path_list, name_list = traverse_dir_files(data_folder)

    yolo = YOLO()

    for count, (path, name) in enumerate(zip(path_list, name_list)):
        # print(path, name)
        try:
            objects_line = yolo.detect_objects_of_image(path)
        except Exception as e:
            print("错误: {}".format(name))
            continue

        if not objects_line:
            continue
        # print(objects_line)
        print("已处理: {}".format(name))
        data_line = '{}---{}'.format(name, objects_line)
        write_line_utf8(out_file, data_line)
        if count % 1000 == 0:
            print('count: {}'.format(count))
        #     break

    yolo.close_session()

    print('执行结束')


if __name__ == '__main__':
    file_name = 'dataset_57w_objects.txt'
    generate_data(file_name)
