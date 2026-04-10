import dlib
import imutils
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QMessageBox, QFileDialog
from pygame import mixer
import cv2
import os
import sys

from imutils import face_utils  # face_utils 模块专门提供了一系列用于处理面部特征的实用函数

# 获取当前文件所在目录的上一级目录，作为基础目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将基础目录添加到系统路径中，以便导入其他模块
sys.path.append(BASE_DIR)

# 从自定义模块ui.UI中导入Ui_MainWindow类
from ui import Ui_MainWindow
# 从自定义模块utils.utils中导入所有内容
from utils import *
import winsound

# 主窗口类，继承自QMainWindow和Ui_MainWindow
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        cv2.waitKey(1)  # 无实际功能，仅用于IDE解析
        # 创建调整摄像头线程对象
        self.adjust_camera_Thread = AdjustCamera_Thread()
        # 创建启动线程对象
        self.start_Thread = Start_Thread()
        # 设置用户界面
        self.setupUi(self)

        # 当摄像头选择框的索引改变时调用change_Cam_Select方法
        self.Cam_Select.currentIndexChanged.connect(self.change_Cam_Select)
        # 当打开视频按钮被点击时调用open_Video方法
        self.Button_OpenVideo.clicked.connect(self.open_Video)
        # 当开始按钮被点击时调用start方法
        self.Button_Start.clicked.connect(self.start)
        # 当结束按钮被点击时调用end方法
        self.Button_End.clicked.connect(self.end)
        # 当调整摄像头位置按钮被点击时调用adjust_camera_location方法
        self.Button_AdjustCamera_Location.clicked.connect(self.adjust_camera_location)
        # 当下班检查复选框状态改变时调用change_OffDuty_Check_Status方法
        self.offDuty_Check.clicked.connect(self.change_OffDuty_Check_Status)
        # 当下班时间值改变时调用change_OffDuty_Value方法
        self.offDuty_Time.valueChanged.connect(self.change_OffDuty_Value)

        # 当视频复选框状态改变时调用set_open_video方法
        self.video.clicked.connect(self.set_open_video)
        # 当摄像头复选框状态改变时调用set_open_video方法
        self.cam.clicked.connect(self.set_open_video)
        # 当显示眼睛复选框状态改变时调用set_show_setting方法
        self.show_eye.clicked.connect(self.set_show_setting)
        # 当显示头部复选框状态改变时调用set_show_setting方法
        self.show_head.clicked.connect(self.set_show_setting)
        # 当显示嘴巴复选框状态改变时调用set_show_setting方法
        self.show_mouth.clicked.connect(self.set_show_setting)
        # 当显示关键点复选框状态改变时调用set_show_setting方法
        self.show_key_point.clicked.connect(self.set_show_setting)

        # 连接start_Thread的msg信号到show_Message槽函数
        self.start_Thread.msg.connect(self.show_Message)
        # 连接start_Thread的picture信号到show_Image槽函数
        self.start_Thread.picture.connect(self.show_Image)
        # 连接start_Thread的window信号到pop_window槽函数
        self.start_Thread.window.connect(self.pop_window)
        # 连接adjust_camera_Thread的picture信号到show_Image槽函数
        self.adjust_camera_Thread.picture.connect(self.show_Image)
        # 连接adjust_camera_Thread的msg信号到show_Message槽函数
        self.adjust_camera_Thread.msg.connect(self.show_Message)
        # 连接adjust_camera_Thread的window信号到pop_window槽函数
        self.adjust_camera_Thread.window.connect(self.pop_window)

    # 设置显示设置的方法
    def set_show_setting(self):
        # 获取发送信号的对象
        isChecked = self.sender().isChecked()
        # 判断发送信号的对象是否为显示眼睛复选框
        if self.sender() == self.show_eye:
            # 设置是否显示眼睛
            self.start_Thread.set_show_eye(isChecked)
        # 判断发送信号的对象是否为显示嘴巴复选框
        elif self.sender() == self.show_mouth:
            # 设置是否显示嘴巴
            self.start_Thread.set_show_mouth(isChecked)
        # 判断发送信号的对象是否为显示头部复选框
        elif self.sender() == self.show_head:
            # 设置是否显示头部
            self.start_Thread.set_show_Head(isChecked)
        else:
            # 设置是否显示关键点
            self.start_Thread.set_show_key_point(isChecked)

    # 设置是否打开视频的方法
    def set_open_video(self):
        # 判断视频复选框是否被选中
        if self.video.isChecked():
            # 设置打开视频
            self.start_Thread.set_open_video(True)
        else:
            # 设置关闭视频
            self.start_Thread.set_open_video(False)

    # 改变下班检查状态的方法
    def change_OffDuty_Check_Status(self):
        # 设置下班检查状态
        self.start_Thread.change_OffDuty_Check_Status(self.offDuty_Check.isChecked())

    # 改变下班时间值的方法
    def change_OffDuty_Value(self):
        # 设置下班时间值
        self.start_Thread.change_OffDuty_Value(self.offDuty_Time.value())

    # 启动线程的方法
    def start(self):
        # 启动start_Thread线程
        self.start_Thread.start()

        # 调整摄像头位置的方法

    def adjust_camera_location(self):
        # 启动adjust_camera_Thread线程
        self.adjust_camera_Thread.start()

        # 结束线程的方法

    def end(self):
        # 关闭adjust_camera_Thread线程
        self.adjust_camera_Thread.close()
        # 关闭start_Thread线程
        self.start_Thread.close()

        # 改变摄像头选择的方法

    def change_Cam_Select(self):
        # 调用adjust_camera_Thread的change_cam_select方法，传入当前选择的摄像头索引
        self.adjust_camera_Thread.change_cam_select(self.Cam_Select.currentIndex())
        # 调用start_Thread的change_cam_select方法，传入当前选择的摄像头索引
        self.start_Thread.change_cam_select(self.Cam_Select.currentIndex())
        # 在输出窗口中添加信息
        self.output_Window.append("切换摄像头" + str(self.Cam_Select.currentIndex()))

        # 打开视频文件的方法

    def open_Video(self):
        # 打开文件对话框，选择视频文件
        filePath = QFileDialog.getOpenFileName(self, "打开视频文件", "", "Video files(*.mp4)")
        # 在输出窗口中添加信息
        self.output_Window.append("视频文件" + filePath[0] + "加载成功")
        # 设置视频文件路径
        self.start_Thread.set_filePath(filePath[0])
        # 选中视频复选框
        self.video.setChecked(True)

        # 显示消息的方法

    def show_Message(self, msg):
        # 在输出窗口中添加消息
        self.output_Window.append(msg)

        # 显示图像的方法

    def show_Image(self, image):
        # 获取图像的高度
        height = image.shape[0]
        # 获取图像的宽度
        width = image.shape[1]
        # 将图像转换为QImage格式
        frame = QImage(image, width, height, QImage.Format_RGB888)
        # 将QImage转换为QPixmap格式
        pix = QPixmap.fromImage(frame)
        # 创建QGraphicsPixmapItem对象
        item = QGraphicsPixmapItem(pix)
        # 创建QGraphicsScene对象
        scene = QGraphicsScene()
        # 将QGraphicsPixmapItem添加到QGraphicsScene中
        scene.addItem(item)
        # 设置图形视图的场景
        self.graphicsView.setScene(scene)

        # 弹出窗口的方法

    def pop_window(self, info):
        # 显示警告消息框
        QMessageBox.warning(self, "提示", info, QMessageBox.Yes)

        # 退出应用程序的方法

    def exit(self):
        # 判断是否有打开的摄像头
        if self.cap is not None:
            # 释放摄像头资源
            self.cap.release()
        # 退出应用程序
        sys.exit(app.exec())

# 调整摄像头线程类，继承自QThread


class AdjustCamera_Thread(QThread):
    # 定义picture信号，用于发送图像数据
    picture = pyqtSignal(object)
    # 定义msg信号，用于发送字符串消息
    msg = pyqtSignal(str)
    # 定义window信号，用于发送字符串信息
    window = pyqtSignal(str)

    def __init__(self):
        super(AdjustCamera_Thread, self).__init__()
        # 初始化面部特征预测器
        self.predictor = None
        # 初始化人脸检测器
        self.detector = None
        # 初始化摄像头对象
        self.cap = None
        # 当前选择的摄像头索引
        self.camSelect = 0
        # 线程关闭标志
        self.isClose = False
        # 加载模型
        self.load_Model()

    # 加载模型的方法
    def load_Model(self):
        # 打印加载面部标志预测器的信息
        print("[日志信息] 加载面部标志预测器...")
        # 使用dlib.get_frontal_face_detector()获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
        # 发送加载成功的消息
        self.msg.emit("脸部特征检测模型加载成功")

    # 改变摄像头选择的方法
    def change_cam_select(self, camSelect):
        # 设置当前选择的摄像头索引
        self.camSelect = camSelect

    # 关闭线程的方法
    def close(self):
        # 设置线程关闭标志为True
        self.isClose = True

    # 线程运行的方法
    def run(self):
        # 设置线程关闭标志为False
        self.isClose = False
        # 发送提示信息
        self.window.emit("请调整摄像头位置，使人脸位于显示框内。调整后请按关闭结束")
        # 打开指定索引的摄像头
        self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
        # 循环读取摄像头画面
        while True:
            # 读取一帧图像
            ret, frame = self.cap.read()
            # 调整图像大小
            frame = imutils.resize(frame, width=720)
            # 将图像转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 自适应直方图均衡
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # 使用detector(gray, 0)进行脸部位置检测
            rects = self.detector(gray, 0)
            # 遍历检测到的人脸
            for rect in rects:
                # 使用predictor(gray, rect)获得脸部特征位置的信息
                shape = self.predictor(gray, rect)
                # 将脸部特征信息转换为数组array的格式
                shape = face_utils.shape_to_np(shape)
                # 获取头部姿态
                reprojectdst, euler_angle = get_head_pose(shape)
                # 取pitch（har）、yaw、roll旋转角度
                pitch = euler_angle[0, 0]
                yaw = euler_angle[1, 0]
                roll = euler_angle[2, 0]

                # 绘制正方体12轴
                for start, end in line_pairs:
                    start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                    end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                # 实时显示计算结果
                cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 0), 2)
                cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

            # 发送图像数据
            self.picture.emit(frame)
            # 判断线程是否关闭
            if self.isClose:
                break

        # 释放摄像头资源
        self.cap.release()
        # 发送摄像头位置调整结束的信息
        self.window.emit("摄像头位置调整结束")

# 启动线程类，继承自QThread
class Start_Thread(QThread):
    # 定义picture信号，用于发送字符串消息
    picture = pyqtSignal(object)
    # 定义msg信号，用于发送字符串消息
    msg = pyqtSignal(str)
    # 定义window信号，用于发送字符串信息
    window = pyqtSignal(str)

    def __init__(self):
        super(Start_Thread, self).__init__()
        # 下班时间
        self.offDutyTime = 0
        # 初始化面部特征预测器
        self.predictor = None
        # 初始化人脸检测器
        self.detector = None
        # 视频文件路径
        self.filePath = None
        # 摄像头对象
        self.cap = None
        # 当前选择的摄像头索引
        self.camSelect = 0
        # 线程关闭标志
        self.isClose = False
        # 是否进行下班检查
        self.isOffDutyCheck = False
        # 是否打开视频
        self.isOpenVideo = False
        # 是否显示眼睛
        self.isShowEye = True
        # 是否显示嘴巴
        self.isShowMouth = True
        # 是否显示头部
        self.isShowHead = False
        # 是否显示关键点
        self.isShowKeyPoint = False
        # 加载模型
        self.load_Model()

        # 加载模型的方法

    def load_Model(self):
        # 打印加载面部标志预测器的信息
        print("[日志信息] 加载面部标志预测器...")
        # 使用dlib.get_frontal_face_detector()获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
        # 发送加载成功的消息
        self.msg.emit("脸部特征检测模型加载成功")

    # 设置是否显示眼睛的方法
    def set_show_eye(self, isShowEye):
        # 设置是否显示眼睛
        self.isShowEye = isShowEye

    # 设置是否显示嘴巴的方法
    def set_show_mouth(self, isShowMouth):
        # 设置是否显示嘴巴
        self.isShowMouth = isShowMouth

    # 设置是否显示头部的方法
    def set_show_Head(self, isShowHead):
        # 设置是否显示头部
        self.isShowHead = isShowHead

    # 设置是否显示关键点的方法
    def set_show_key_point(self, isShowKeyPoint):
        # 设置是否显示关键点
        self.isShowKeyPoint = isShowKeyPoint

    # 改变下班检查状态的方法
    def change_OffDuty_Check_Status(self, isOffDutyCheck):
        # 设置是否进行下班检查
        self.isOffDutyCheck = isOffDutyCheck

    # 改变下班时间值的方法
    def change_OffDuty_Value(self, offDutyTime):
        # 设置下班时间值
        self.offDutyTime = offDutyTime

    # 改变摄像头选择的方法
    def change_cam_select(self, camSelect):
        # 设置当前选择的摄像头索引
        self.camSelect = camSelect

    # 设置视频文件路径的方法
    def set_filePath(self, filePath):
        # 设置是否打开视频
        self.isOpenVideo = True
        # 设置视频文件路径
        self.filePath = filePath

    # 设置是否打开视频
    def set_open_video(self, isOpenVideo):
        self.isOpenVideo = isOpenVideo
    # 关闭线程的方法
    def close(self):
        self.isClose = True

    # 播放警告音乐的静态方法
    @staticmethod
    def playMusic():
        mixer.init()
        mixer.music.load('media/Tired.mp3')
        mixer.music.play()

    def run(self):
        # 发送开始程序的提示信息
        self.window.emit("开始程序")
        # 设置线程关闭标志为False
        self.isClose = False

        # 打开相机/视频
        if self.isOpenVideo:
            # 若未指定视频文件路径，提示用户并返回
            if self.filePath is None:
                self.window.emit("未加载视频，请加载视频后再点击")
                return
            # 打开指定的视频文件
            self.cap = cv2.VideoCapture(self.filePath)
            self.msg.emit("视频读取成功")
        else:
            # 打开指定索引的摄像头
            self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
            self.msg.emit("相机打开成功")

        # 初始化数据参数（测试次数、测试EAR、MAR和HAR的和、测试次数魔法值）
        test_time = 0
        TEST_TIMES = 100
        ear_sum = 0
        mar_sum = 0
        har_sum = 0

        Detected_TIME_LIMIT = 60
        closed_times = 0
        yawning_times = 0
        pitch_times = 0
        warning_time = 0

        # 阈值（EAR、MAR、HAR、per clos阈值）
        EAR_THRESH = 0
        MAR_THRESH = 0
        HAR_THRESH = 0
        FATIGUE_THRESH = 0.4
        PITCH_THRESH = 6
        offDutyTime = 0

        self.msg.emit("程序正在计算面部特征阈值，请您耐心等待")
        self.window.emit("程序正在计算面部特征阈值，请您耐心等待")

        # 从视频流循环帧
        while True:
            # 读取一帧图像
            ret, frame = self.cap.read()
            # 若读取失败，判断是否为视频播放结束
            if not ret:
                if self.isOpenVideo:
                    self.window.emit("视频播放结束")
                    print("视频结束")
                break

            # 调整图像大小
            frame = imutils.resize(frame, width=720)
            # 将图像转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 限制对比度自适应直方图均衡
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # 使用检测器进行脸部位置检测
            rects = self.detector(gray, 0)

            # 脱岗检测
            if not rects:
                if self.isOffDutyCheck:
                    offDutyTime += 1
                    # 若脱岗时间超过设定值，提示用户
                    if offDutyTime >= self.offDutyTime * 30:
                        self.msg.emit("您已经脱岗，请立刻回到岗位")
                        offDutyTime = 0
            else:
                offDutyTime = 0

            # 面部特征检测
            for rect in rects:
                # 获取脸部特征位置的信息
                shape = self.predictor(gray, rect)

                if self.isShowKeyPoint:
                    # 获取关键点的坐标
                    for point in shape.parts():
                        # 每个点的坐标
                        point_position = (point.x, point.y)
                        # 绘制关键点
                        cv2.circle(frame, point_position, 3, (255, 8, 0), -1)

                # 将脸部特征信息转换为数组array的格式
                shape = face_utils.shape_to_np(shape)
                # 提取左眼、右眼坐标、嘴巴坐标
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]

                # 计算左右眼的EAR平均值、计算嘴巴MAR值
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)

                # 获取头部姿态
                reprojectdst, euler_angle = get_head_pose(shape)
                # 取pitch（har）、yaw、roll旋转角度
                pitch = euler_angle[0, 0]
                yaw = euler_angle[1, 0]
                roll = euler_angle[2, 0]
                har = pitch

                if self.isShowHead:
                    # 绘制正方体12轴
                    for start, end in line_pairs:
                        start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                        end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                # 实时显示计算结果
                cv2.putText(frame, "ear: {}".format(ear), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "mar: {}".format(mar), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 计算100次ear、mar和har数据求平均值，得到当前使用者眼部、嘴部、头部俯仰的阈值
                if test_time < TEST_TIMES:
                    test_time += 1
                    ear_sum += ear
                    mar_sum += mar
                    har_sum += har

                    if test_time == TEST_TIMES:
                        EAR_THRESH = ear_sum / TEST_TIMES
                        MAR_THRESH = mar_sum / TEST_TIMES
                        HAR_THRESH = har_sum / TEST_TIMES
                        print('眼睛长宽比ear 100次取平均的阈值:{:.2f}'.format(EAR_THRESH))
                        print('嘴部长宽比mar 100次取平均的阈值:{:.2f}'.format(MAR_THRESH))
                        print('头部俯仰角pitch 100次取平均的阈值:{:.2f}'.format(HAR_THRESH))
                        self.msg.emit('眼睛长宽比ear 100次取平均的阈值:{:.2f}'.format(EAR_THRESH))
                        self.msg.emit('嘴部长宽比mar 100次取平均的阈值:{:.2f}'.format(MAR_THRESH))
                        self.msg.emit('头部俯仰角pitch 100次取平均的阈值:{:.2f}'.format(HAR_THRESH))
                        continue

                # 画图,嘴巴、眼睛凸包标注,用矩形框标注人脸
                if self.isShowEye:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if self.isShowMouth:
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

                """
                    计算检测时间内，异常状态的次数
                    异常状态定义:
                    1.EAR小于0.8标记为闭合
                    2.MAR大于1.5倍即异常
                    3.HAR跟阈值差大于标准值

                    异常状态判断解释：
                    在每次循环中，根据眼睛长宽比 ear、嘴巴长宽比 mar 和头部俯仰角 har 与相应阈值的比较结果，增加对应的异常状态次数。
                    例如，当 ear < 0.75 * EAR_THRESH 时，增加 closed_times 的值；
                    当 mar > 1.6 * MAR_THRESH 时，增加 yawning_times 的值；
                    当 abs(har - HAR_THRESH) > PITCH_THRESH 时，增加 pitch_times 的值
                """
                if Detected_TIME_LIMIT > 0:
                    Detected_TIME_LIMIT -= 1
                    if ear < 0.75 * EAR_THRESH:
                        closed_times += 1
                    if mar > 1.5 * MAR_THRESH:
                        yawning_times += 1
                    if abs(har - HAR_THRESH) > PITCH_THRESH:
                            pitch_times += 1
                else:
                    # 重置检测时间限制
                    Detected_TIME_LIMIT = 60
                    isEyeTired = False
                    isYawnTired = False
                    isHeadTired = False

                    # 判断是否疲劳,大于阈值则疲劳
                    if closed_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("闭眼时长较长")
                        isEyeTired = True

                    if yawning_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("张嘴时长较长")
                        isYawnTired = True

                    if pitch_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                        self.msg.emit("低头时长较长")
                        isHeadTired = True

                    # 重置次数
                    closed_times = 0
                    yawning_times = 0
                    pitch_times = 0

                    isWarning = False
                    # 疲劳状态判断
                    if isEyeTired and isYawnTired:
                        warning_time += 2
                        isWarning = True
                    elif isHeadTired and isEyeTired:
                        warning_time += 2
                        isWarning = True
                    elif isEyeTired:
                        warning_time += 1
                        isWarning = True
                    elif isYawnTired:
                        warning_time += 1
                        isWarning = True
                    elif isHeadTired:
                        warning_time += 1
                        isWarning = True
                    else:
                        warning_time = 0

                    # 当 warning_time >= 3 时，不仅发出声音提醒（通过 self.playMusic() 播放更完整的警告音频），
                    # 还会在界面上显示相应的疲劳提示信息。
                    if warning_time >= 3:
                        warning_time = 0
                        self.msg.emit("您已经疲劳，请注意休息")
                        self.window.emit("您已经疲劳，请注意休息")
                        self.playMusic()
                    else:
                        # 异常动作，发出发出频率为 440 赫兹、持续时间为 1000 毫秒（即 1 秒）的声音提醒
                        if isWarning:
                            winsound.Beep(440, 1000)


            # 发送图像数据用于显示
            self.picture.emit(frame)

            # 判断线程是否关闭
            if self.isClose:
                break

        # 释放摄像头或视频资源
        self.cap.release()

if __name__ == '__main__':
    # 创建Qt应用程序对象
    app = QtWidgets.QApplication(sys.argv)
    # 创建主窗口对象
    mainWindow = MainWindow()
    # 显示主窗口
    mainWindow.show()
    # 进入应用程序的主循环
    sys.exit(app.exec())

