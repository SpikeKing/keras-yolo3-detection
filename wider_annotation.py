#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/6/14
"""
import os

val_bbx_file = 'dataset/wider_face_split/wider_face_val_bbx_gt.txt'
train_bbx_file = 'dataset/wider_face_split/wider_face_train_bbx_gt.txt'

val_data_folder = 'dataset/WIDER_val'
train_data_folder = 'dataset/WIDER_train'

out_file = 'dataset/WIDER_train.txt'


def generate_train_file(bbx_file, data_folder, out_file):
    paths_list, names_list = traverse_dir_files(data_folder)
    name_dict = dict()
    for path, name in zip(paths_list, names_list):
        name_dict[name] = path

    data_lines = read_file(bbx_file)

    sub_count = 0
    item_count = 0
    out_list = []

    for data_line in data_lines:
        item_count += 1
        if item_count % 1000 == 0:
            print('item_count: ' + str(item_count))

        data_line = data_line.strip()
        l_names = data_line.split('/')
        if len(l_names) == 2:
            if out_list:
                out_line = ' '.join(out_list)
                write_line(out_file, out_line)
                out_list = []

            name = l_names[-1]
            img_path = name_dict[name]
            sub_count = 1
            out_list.append(img_path)
            continue

        if sub_count == 1:
            sub_count += 1
            continue

        if sub_count >= 2:
            n_list = data_line.split(' ')
            x_min = n_list[0]
            y_min = n_list[1]
            x_max = str(int(n_list[0]) + int(n_list[2]))
            y_max = str(int(n_list[1]) + int(n_list[3]))
            p_list = ','.join([x_min, y_min, x_max, y_max, '0'])  # 标签全部是0，人脸
            out_list.append(p_list)
            continue


def traverse_dir_files(root_dir, ext=None):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param ext: 后缀名
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def sort_two_list(list1, list2):
    """
    排序两个列表
    :param list1: 列表1
    :param list2: 列表2
    :return: 排序后的两个列表
    """
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def read_file(data_file, mode='more'):
    """
    读文件, 原文件和数据文件
    :return: 单行或数组
    """
    try:
        with open(data_file, 'r') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                # return map(str.strip, output)
                return output
            else:
                return list()
    except IOError:
        return list()


def write_line(file_name, line):
    """
    将行数据写入文件
    :param file_name: 文件名
    :param line: 行数据
    :return: None
    """
    if file_name == "":
        return
    with open(file_name, "a+") as fs:
        if type(line) is (tuple or list):
            fs.write("%s\n" % ", ".join(line))
        else:
            fs.write("%s\n" % line)


if __name__ == '__main__':
    generate_train_file(val_bbx_file, val_data_folder, out_file)
    generate_train_file(train_bbx_file, train_data_folder, out_file)
