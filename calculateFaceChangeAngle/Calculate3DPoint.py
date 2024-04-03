# import numpy as np
# import cv2
#
# # 旋转向量和平移向量
# rotation_vector = np.array([[2.82915254], [0.1152089], [0.84478683]])
# translation_vector = np.array([[431.57348881], [394.83388037], [2379.54565124]])
#
# # 将旋转向量转换为旋转矩阵
# rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#
# # 点P在相机坐标系下的三维坐标
# point_in_camera_coord = np.array([(0.0, 0.0, 1000.0)])
#
# # 将点P由相机坐标系转换到世界坐标系
# point_in_world_coord = rotation_matrix.dot(point_in_camera_coord.T) + translation_vector
#
# print("Point in World Coordinate System:\n {}".format(point_in_world_coord))
# # Point in World Coordinate System:
# #  [[ 981.5318934 ]
# #  [ 239.15680969]
# #  [1558.9907153 ]]

import numpy as np
import cv2


# 这个函数接受旋转向量和平移向量作为输入，并返回旋转和平移后的点坐标
def transform_point(rotation_vector, translation_vector, point):
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)


    # 将点坐标转换为齐次坐标（增加一个维度以便应用平移）
    point_homogeneous = np.concatenate((point, np.array([[1]])), axis=1).T

    # 创建变换矩阵（旋转和平移）
    transform_matrix = np.hstack((rotation_matrix, translation_vector))
    transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))

    # 应用变换矩阵到点坐标
    point_transformed_homogeneous = np.dot(transform_matrix, point_homogeneous)

    # 将点的齐次坐标转换回普通坐标
    point_in_world_coord = point_transformed_homogeneous[:3].T
    # point_in_world_coord = point_transformed_homogeneous

    print("Transformed Point:\n {}".format(point_in_world_coord))
    return point_in_world_coord


if __name__ == "__main__":
    # 给定的旋转向量和平移向量
    rotation_vector = np.array([[2.82915254], [0.1152089], [0.84478683]])
    translation_vector = np.array([[431.57348881], [394.83388037], [2379.54565124]])

    # 定义点的坐标
    point = np.array([(0.0, 0.0, 1000.0)])

    # 计算变换后的点的坐标
    transformed_point = transform_point(rotation_vector, translation_vector, point)