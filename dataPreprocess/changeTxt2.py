# 用这个改变YOLOv5的txt文件中的label编号
# 更改text_file_path目录里的所有txt文件，用input_string代替第一个输入的字符串

import os
import re

def modify_txt(text_file_path, input_string):
    text_files = os.listdir(text_file_path)

    for text_file in text_files:
        print(text_file)
        with open(os.path.join(text_file_path, text_file), 'r+') as f:
            old_line = f.read()
            lines = old_line.splitlines()
            line_new = ""
            for line in lines:
                # split()方法的第二个参数设置为1，这意味着它只会在第一个空格处进行一次分割。
                split_strings = line.split(" ", 1)

                string1 = split_strings[0]
                string2 = split_strings[1]
                # 有个弊端，最后一行也多了一个\n
                line_new += input_string+" "+string2 +"\n"

            print(old_line)
            print("change to")
            print(line_new)
            # print(old_line)
            print()
            f.seek(0)  # 将指针位置指到文件开头
            f.truncate()  # 清空文件内容
            f.write(line_new)

if __name__ == '__main__':
    # modify_txt('./test/', '1')# for test
    # modify_txt('./pen/Pen/train/labels/', "80")
    # modify_txt('./mouse/mouse/train/labels/', "64")
    modify_txt('./book/book/train/labels/', "73")