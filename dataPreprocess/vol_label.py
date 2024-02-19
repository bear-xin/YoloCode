# -*- coding: utf-8 -*-
# 使用这个文件之前要先用split_train_val.py文件划分数据集、得到ImageSet里面每个数据集的文件名称
# 该文件用于获取val/test等数据集的文件绝对路径，如果倒数第二行开了，还可以实现xml到txt的label文件转化
# 这两个文件都应该在root/../datasets/里面使用，只是放入YoloCode备份一下
# 使用前记得把文件后缀检查一下.jpg/.png

import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ["cat", "dog"]   # 改成自己的类别
abs_path = os.getcwd()
print(abs_path)

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id):
    # 用相对路径方式设置输入输出文件
    in_file_path = os.path.join(abs_path, 'annotations/%s.xml'% (image_id))
    # print(in_file_path)
    # in_file = open('/home/trainingai/zyang/yolov5/paper_data/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    in_file = open(in_file_path, encoding='UTF-8')
    out_file_path = os.path.join(abs_path, 'labels/%s.txt'% (image_id))
    # out_file = open('/home/trainingai/zyang/yolov5/paper_data/labels/%s.txt' % (image_id), 'w')
    out_file = open(out_file_path, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # print(w,h)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # print(b1, b2, b3, b4)
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

"""对三个数据集进行操作"""
# abs_path 代表 当前目录的绝对路径
for image_set in sets:
    # 相对路径
    rel_path = 'labels/'
    # 完整路径
    labels_path = os.path.join(abs_path, rel_path)
    # 没有labels文件夹创建它
    # if not os.path.exists('/home/trainingai/zyang/yolov5/paper_data/labels/'):
    #     os.makedirs('/home/trainingai/zyang/yolov5/paper_data/labels/')
    #
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    # 获取该数据集文件名称列表所在的文件路径
    image_ids_path = os.path.join(abs_path, 'ImageSets/%s.txt' % (image_set))
    # 获取该数据集文件名称列表
    # image_ids = open('/home/trainingai/zyang/yolov5/paper_data/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    image_ids = open(image_ids_path).read().strip().split()
    # 打开最终存储 绝对路径数据集文件名称列表 文件，成为list_file
    list_file = open('%s.txt' % (image_set), 'w')
    # 把每张图片的绝对路径都写入list_file中
    for image_id in image_ids:
        list_file.write(abs_path + '/images/%s.png\n' % (image_id))
        # 开启这句就是要转换xml文件到txt文件，不开就是不做更多事情了
        convert_annotation(image_id)
    list_file.close()
