'''
使用：在main中修改参数，先后调用两个函数
用途：用于在yolov5格式的数据集中提取出包含某一个类型的图片，形成新的数据集
'''
import os
import shutil

'''
从source_dir目录下读取所有的txt文件，如果该文件的label包含kind类，则将该txt文件复制到target_dir目录下，从而筛选出所有包含kind类的文件
'''
def FisrtStep(source_dir, target_dir, kinds):
    # 获取指定目录下的所有txt文件
    txt_files = [file for file in os.listdir(source_dir) if file.endswith('.txt')]

    for txt_file in txt_files:
        with open(os.path.join(source_dir, txt_file), 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 划分每一行的内容
                elements = line.strip().split()
                if elements[0] in kinds: # keyboard
                    # 复制符合条件的txt文件到目标目录
                    shutil.copyfile(os.path.join(source_dir, txt_file), os.path.join(target_dir, txt_file))
                    break  # 如果找到符合条件的行，则跳出当前txt文件的循环

'''
从txt_dir目录下读取所有的txt文件名，找到jpg_dir目录下与其的重名的.jpg文件，复制到target_dir目录下
'''
def SecondSteo(txt_dir, jpg_dir, target_dir):

    # 获取txt目录下的所有txt文件名（不带扩展名）
    txt_files = [os.path.splitext(file)[0] for file in os.listdir(txt_dir) if file.endswith('.txt')]

    for txt_file in txt_files:
        jpg_file = txt_file + '.jpg'
        jpg_path = os.path.join(jpg_dir, jpg_file)
        target_path = os.path.join(target_dir, jpg_file)

        if os.path.isfile(jpg_path):
            # 复制jpg文件到目标目录
            shutil.copyfile(jpg_path, target_path)
            print(f"复制成功：{jpg_file}")
        else:
            print(f"未找到对应的jpg文件：{jpg_file}")

'''在此处调用上面两个步骤，安全又保险'''
if __name__ == "__main__":

    source_dir = './COCOtest/'
    target_dir = './COCOans/labels/'
    kinds = ['66', '67', '68', '69']
    FisrtStep(source_dir, target_dir, kinds)

    txt_dir = './COCOans/labels/'
    jpg_dir = './coco/val2017/images'
    target_dir = './COCOans/images/'
    SecondSteo(txt_dir, jpg_dir, target_dir)