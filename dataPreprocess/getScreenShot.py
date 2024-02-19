import cv2 as cv
def video_demo():
    capture =cv.VideoCapture(0)
    # 获取视频帧的原始宽度和高度。
    width, height =capture.get(3),capture.get(4)
    # 将视频帧的宽度和高度设置为原始值的1.5倍。可以看的更广。
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width* 2)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height* 2)
    index = 0
    while True:
        # ：读取视频流的下一帧。ret是一个布尔值，表示是否成功读取到了一帧。frame是一个表示视频帧的图像数组。
        ret,frame = capture.read()
        # 水平翻转视频帧，使其呈现镜像效果。
        frame = cv.flip(frame,1)
        # 在名为"video"的窗口中显示当前的视频帧。
        cv.imshow("video", frame)
        # 按下's'键时，会保存当前帧为图像文件。
        if cv.waitKey(1) == ord('s'):
            # 将当前视频帧保存为图像文件，文件名为索引加上.jpg后缀，保存在当前目录下的image文件夹中。
            cv.imwrite("../image/"+str(index)+".jpg",frame)
            print("../image/"+str(index)+".jpg")
            index+=1
        # 按下ESC键时，程序退出并关闭窗口。
        if cv.waitKey(1)== 27:
            cv.destroyAllWindows()
            break
if __name__=="__main__":
    video_demo()