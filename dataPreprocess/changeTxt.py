# 用这个改变YOLOv5的txt文件中的label编号
# 更改text_file_path目录里的所有txt文件，用output_string代替input_string

import os
import re

def modify_txt(text_file_path, input_string, output_string):
    text_files = os.listdir(text_file_path)

    for text_file in text_files:
        print(text_file)
        with open(os.path.join(text_file_path, text_file), 'r+') as f:
            old_line = f.read()
            line_new = re.sub(input_string, output_string, old_line, 1)
            print(old_line)
            print(line_new)
            print()
            f.seek(0)  # 将指针位置指到文件开头
            f.truncate()  # 清空文件内容
            f.write(line_new)

if __name__ == '__main__':
    modify_txt('./earphone/EarphoneDetection/bear/labels/', '1', '0')