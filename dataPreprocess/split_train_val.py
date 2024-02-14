# coding:utf-8
# 先用这个py划分数据集
# --xml_path：获得图片名称集合的文件夹（可以不是xml文件，因为只需要获得该文件夹下的文件和文件夹名称；
#                                   但需要注意，该名称不包含'.xml'，通过忽略名称后三位实现，如果是'.doxc'等不等于三位的，要改一下位数）
# --txt_path：四个集合输出的地方
# 这两个文件都应该在root/../datasets/里面使用，只是放入YoloCode备份一下

import os
import random
import argparse

"""指定xml文件输入地址、文件名称输出地址"""
#命令行解析器的类，可以定义命令行参数及其相关配置
parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在annotations下
parser.add_argument('--xml_path', default='annotations', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='ImageSets', type=str, help='output txt label path')
opt = parser.parse_args()

"""指定两个集合的划分概率"""
trainval_percent = 1
train_percent = 0.9

# 过 opt.xml_path 来获取 --xml_path 参数的值
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
# 获取指定目录中所有文件和文件夹的列表
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
num = len(total_xml)
list_index = range(num)
# 获取val集合的图片个数
tv = int(num * trainval_percent)
# 获取train集合的图片个数
tr = int(tv * train_percent)
# 随机获取val集合编号
trainval = random.sample(list_index, tv)
# 随机获取train集合编号
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

# 根据图片的编号是否在val等集合中,来判断如何写入内容
for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
