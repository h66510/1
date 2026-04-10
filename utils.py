"""
@author:兵慌码乱
@project_name:Dlib疲劳检测警报系统
@time:2025/7/27
@remarks:工具函数

1.定义3D参考点和相机参数：定义了人脸上特定部位在3D空间中的位置，以及相机的内参和畸变系数
2.计算头部姿态：通过solvePnP函数，根据3D参考点和2D特征点，计算出头部的旋转和平移矩阵，并计算欧拉角
3.计算眼睛和嘴部的长宽比：通过计算眼睛和嘴部特征点之间的距离，得到眼睛和嘴部的长宽比
"""
# 导入必要的包
import math  # 用于数学计算
import cv2  # 用于计算机视觉任务，如图像处理和视频处理
import numpy as np  # 数据处理的库 numpy
from imutils import face_utils  # 用于处理面部特征点
from scipy.spatial import distance as dist  # 用于计算两点之间的距离

# 点代表了人脸上特定部位在3D空间中的位置
object_pts = np.float32([[6.825897,6.760612,4.402142],  # 33左眉左上角
                         [1.330353,7.122144,6.903745],  # 29左眉右角
                         [-1.330353,7.122144,6.903745],  # 34右眉左角
                         [-6.825897,6.760612,4.402142],  # 38右眉右上角
                         [5.311432,5.485328,3.987654],  # 13左眼左上角
                         [1.789930,5.393625,4.413414],  # 17左眼右上角
                         [-1.789930,5.393625,4.413414],  # 25右眼左上角
                         [-5.311432,5.485328,3.987654],  # 21右眼右上角
                         [2.005628,1.409845,6.165652],  # 55鼻子左上角
                         [-2.005628,1.409845,6.165652],  # 49鼻子右上角
                         [2.774015,-2.080775,5.048531],  # 43嘴左上角
                         [-2.774015,-2.080775,5.048531],  # 39嘴右上角
                         [0.000000,-3.116408,6.097667],  # 45嘴中央下角
                         [0.000000,-7.415691,4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
# 相机内参矩阵描述了相机的内部特性，如焦距、主点位置等
K = [6.5308391993466671e+002,0.0,3.1950000000000000e+002,
     0.0,6.5308391993466671e+002,2.3950000000000000e+002,
     0.0,0.0,1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
# 相机畸变参数用于校正相机镜头产生的畸变
D = [7.0834633684407095e-002,6.9140193737175351e-002,0.0,0.0,-1.3073460323689292e+000]
# 像素坐标系(xy)：填写凸轮的本征和畸变系数
# 将相机内参和畸变系数转换为numpy数组
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
# 这些点用于在计算出头部姿态后，重新投影到图像平面上，以验证结果的准确性
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
# 这些线对用于连接重新投影后的3D点，形成一个正方体，以便可视化头部姿态
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# 第三步：分别获取左右眼面部标志的索引
# 从face_utils中获取左右眼和嘴部特征点的起始和结束索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 头部姿态估计
def get_head_pose(shape):
    # 这些点是人脸上在图像平面上的2D特征点，与3D参考点对应
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应
    # 通过solvePnP函数，根据3D参考点和2D特征点，计算出头部的旋转和平移矩阵
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    # 将3D参考点重新投影到图像平面上，得到重投影后的2D点
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 将旋转向量转换为旋转矩阵
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat是垂直拼接
    # 将投影矩阵分解为旋转矩阵和相机矩阵，并计算欧拉角
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    # 将欧拉角从强度转换为角度
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    # 对欧拉角进行一些修正
    pitch = math.degrees(math.asin(math.sin(pitch)))  # 俯仰角
    """
    Yaw（偏航角）：表示头部绕垂直轴（Y轴）的左右旋转。
    正值：头部向左转（从观察者视角看，人脸偏向左侧）。
    负值：头部向右转（从观察者视角看，人脸偏向右侧）。
    应用场景：检测驾驶员是否在左右张望（如看后视镜或侧窗），可能导致注意力分散。
    """
    roll = -math.degrees(math.asin(math.sin(roll)))
    """
    Roll（翻滚角）：表示头部绕前后轴（Z轴）的倾斜。
    正值：头部向左侧倾斜（左耳靠近左肩）。
    负值：头部向右侧倾斜（右耳靠近右肩）。
    应用场景：检测头部是否倾斜（如疲劳时无意识歪头），或姿势异常。
    """
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return reprojectdst, euler_angle  # 返回投影误差，欧拉角

# 眼部姿态估计
def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    # 计算垂直方向上眼睛特征点之间的欧式距离
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear



# 嘴部姿态估计
def mouth_aspect_ratio(mouth):# 嘴部
    # 计算嘴部特征点之间的欧几里得距离
    A = np.linalg.norm(mouth[2] - mouth[9]) # 51, 58
    B = np.linalg.norm(mouth[4] - mouth[7]) # 53, 56
    C = np.linalg.norm(mouth[0] - mouth[6]) # 49, 55
    # 计算嘴部的长宽比
    mar = (A + B) / (2.0 * C)
    return mar