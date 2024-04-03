import cv2
import os

# 创建保存图片的文件夹（如果不存在）
save_folder = './image/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 初始化摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

while True:
    # 读取当前帧
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from video device.")
        break

    # 显示当前帧
    cv2.imshow('Video Feed', frame)

    # 等待按键事件
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 's' 键，保存图像
    if key == ord('s'):
        # 检查当前文件夹中文件数量
        files = [f for f in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, f))]
        image_count = len(files)

        # 构造新文件名
        image_name = f"image{image_count + 1}.jpg"
        image_path = os.path.join(save_folder, image_name)

        # 保存图像
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")

    # 如果按下 'q' 键，退出循环
    elif key == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()