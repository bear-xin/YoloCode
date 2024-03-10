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
def SecondStep(txt_dir, jpg_dir, target_dir):

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

'''

'''
def ThreeStep(txt_dir, ans_dir, kinds):
    # 创建输出目录
    os.makedirs(ans_dir, exist_ok=True)

    # 获取txt目录下的所有txt文件
    txt_files = [file for file in os.listdir(txt_dir) if file.endswith('.txt')]

    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)

        # 读取txt文件内容
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        # 筛选保留符合要求的行
        filtered_lines = [line for line in lines if line.split(' ')[0] in kinds]

        # 写入结果到输出目录下的同名文件
        ans_path = os.path.join(ans_dir, txt_file)
        with open(ans_path, 'w') as file:
            file.writelines(filtered_lines)

        print(f"已处理：{txt_file}")

    print("筛选完成！")

'''在此处调用上面步骤，安全又保险'''
if __name__ == "__main__":
    '''
    0:person
    39:cup
    41:cup
    62:TV
    63:laptop
    64:mouse
    65:remote
    66:keyboard
    67:cell phone
    73:book
    '''
    source_dir = './COCOtest/'
    target_dir = './COCOans/midAns/'
    kinds = ['39', '41', '62', '63', '64', '66', '67', '73']
    FisrtStep(source_dir, target_dir, kinds)

    txt_dir = './COCOans/midAns/'
    jpg_dir = './coco/val2017/images'
    target_dir = './COCOans/images/'
    SecondStep(txt_dir, jpg_dir, target_dir)

    txt_dir = './COCOans/midAns/'
    ans_dir = './COCOans/labels/'
    kinds = ['39', '41', '62', '63', '64', '66', '67', '73']  # 需要匹配的字符串列表
    ThreeStep(txt_dir, ans_dir, kinds)