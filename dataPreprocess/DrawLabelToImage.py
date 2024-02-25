# 根据image_dir下的.jpg文件及label_dir文件夹下与其名称相同符合yolov5格式的.txt的标签文件，输出每张图片对应的被标签好图片到output_dir目录。
# 其中有num_classes个类，对于每一种类都用固定的一个颜色，每个类颜色不同。
import os
import cv2
import random
import matplotlib.pyplot as plt

image_dir = './images/'
label_dir = './labels/'
output_dir = './labeled_images/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义颜色映射表
color_map = {}
num_classes = 80  # 类的总数

# 生成不同的颜色，并与类进行映射
for i in range(num_classes):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color_map[i] = color

# 获取.image目录下的所有.jpg文件名（不带扩展名）
image_files = [os.path.splitext(file)[0] for file in os.listdir(image_dir) if file.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file + '.jpg')
    label_path = os.path.join(label_dir, image_file + '.txt')
    output_path = os.path.join(output_dir, image_file + '_labeled.jpg')

    if os.path.isfile(label_path):
        # 读取原始图像
        image = cv2.imread(image_path)

        # 读取标签文件内容
        with open(label_path, 'r') as file:
            lines = file.readlines()

        # 处理每一行标签
        for line in lines:
            line = line.strip().split()
            class_id = int(line[0])
            x_center = float(line[1]) * image.shape[1]
            y_center = float(line[2]) * image.shape[0]
            width = float(line[3]) * image.shape[1]
            height = float(line[4]) * image.shape[0]

            # 计算边界框的坐标
            x_min = int(x_center - width/2)
            y_min = int(y_center - height/2)
            x_max = int(x_center + width/2)
            y_max = int(y_center + height/2)

            # 绘制边界框和类别标签
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_map[class_id], 2)
            cv2.putText(image, f'{class_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[class_id], 2)

        # 保存标签好的图像
        cv2.imwrite(output_path, image)
        print(f"已完成标签绘制：{image_file}.jpg")
    else:
        print(f"未找到对应的标签文件：{image_file}.txt")