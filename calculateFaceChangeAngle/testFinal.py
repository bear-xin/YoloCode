import numpy as np

from Calculate3DPoint import transform_point
from CopyOthers_test import getRotAndRev


# 定义点的坐标
pointNose = np.array([(0.0, 0.0, 0.0)])
pointNoseEnd = np.array([(0.0, 0.0, 1000.0)])

rotation_vector1, translation_vector1 = getRotAndRev('./image/image1.jpg')
transformed_pointNose1 = transform_point(rotation_vector1, translation_vector1, pointNose)
transformed_pointNoseEnd1 = transform_point(rotation_vector1, translation_vector1, pointNoseEnd)

rotation_vector2, translation_vector2 = getRotAndRev('./image/image3.jpg')
transformed_pointNose2 = transform_point(rotation_vector2, translation_vector2, pointNose)
transformed_pointNoseEnd2 = transform_point(rotation_vector2, translation_vector2, pointNoseEnd)

# print(type(transformed_pointNose1))


# 计算向量AB和CD
AB = transformed_pointNoseEnd1 - transformed_pointNose1
CD = transformed_pointNoseEnd2 - transformed_pointNose2

# 将向量 A, B, C, D 转换为1D数组
AB = AB.flatten()
CD = CD.flatten()


# 计算点积（向量点乘）
dot_product = np.dot(AB, CD)  # 这里不会有形状不一致的错误

# 计算向量 AB 和 CD 的模长
magnitude_AB = np.linalg.norm(AB)
magnitude_CD = np.linalg.norm(CD)

# 计算夹角的余弦值
cos_angle = dot_product / (magnitude_AB * magnitude_CD)

# 计算夹角（弧度制）
angle_radians = np.arccos(cos_angle)

# 如果需要，将弧度转为度
angle_degrees = np.degrees(angle_radians)

print("Angle in radians: ", angle_radians)
print("Angle in degrees: ", angle_degrees)