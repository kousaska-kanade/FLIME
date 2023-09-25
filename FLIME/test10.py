# human_face.py
import cv2

# 读入图像
img = cv2.imread(r'D:\PyCharm 2021.1.3\pycharmproject\Lime-pytorch\data\20-4.JPG')  # 读入 ,放在同目录

# 检测图像中的人脸
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 创建人脸检测器    放在同目录
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将img转为回复图像，存放中gray中
faces = face.detectMultiScale(gray, 1.1, 3)  # 检测图像中的人脸
for (x, y, w, h) in faces:  # 标注人脸区域
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)

cv2.imshow('result', img)  # 显示
cv2.waitKey(0)  # 按任意键退出
cv2.destroyAllWindows()  # 关闭所有窗口
