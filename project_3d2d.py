import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from bounding_box_3d.msg import BoundingBox3D
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import tf
import pandas
import matplotlib.pyplot as plt 
import torch

from sensor_msgs import point_cloud2 as pc2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort.sort.detection import Detection
from deep_sort.deep_sort.sort.tracker import Tracker
from deep_sort.deep_sort.deep_sort_lidar import DeepSort
# from deep_sort import generate_detections
from deep_sort.deep_sort.sort import preprocessing as prep

# from tracker.msg import TrackerObject, TrackerObjectArray
# # 相机内参矩阵
# K = np.array([9.597910e+02, 0.000000e+00, 6.960217e+02],
#     [0.000000e+00, 9.569251e+02, 2.241806e+02],
#     [0.000000e+00, 0.000000e+00, 1.000000e+00])

# # 相机畸变参数
# D = np.array([-0.3728755, 0.2037299, 0.002219027, 0.001383707, -0.07233722])

# # lidar 到相机的外参矩阵（旋转矩阵 R 和平移向量 T）
# R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
#               [1.480249e-02, 7.280733e-04, -9.998902e-01],
#               [9.998621e-01, 7.523790e-03, 1.480755e-02]])
# T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

class Detector: 

    def __init__(self):
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        self._bridge = CvBridge()
        self.count = 0
        self.deep_weights = '/home/sam/iv_ws/src/tracker/src/deep_sort/model_data/mars-small128.pb'
        # 相机内参矩阵 K
        self.K = np.array([
            [9.597910e+02, 0.000000e+00, 6.960217e+02],
            [0.000000e+00, 9.569251e+02, 2.241806e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])

        # 相机畸变系数 D
        self.D = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])

        # LiDAR 到相机 00 的外参
        R1 = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04],
            [1.480249e-02, 7.280733e-04, -9.998902e-01],
            [9.998621e-01, 7.523790e-03, 1.480755e-02]
        ])
        T1 = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        # 相机 00 到相机 02 的外参
        R2 = np.array([
            [9.999758e-01, -5.267463e-03, -4.552439e-03],
            [5.251945e-03, 9.999804e-01, -3.413835e-03],
            [4.570332e-03, 3.389843e-03, 9.999838e-01]
        ])
        T2 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])

        # 计算变换矩阵
        T_final = np.dot(R2, np.hstack((R1, T1.reshape(-1, 1)))) + T2.reshape(-1, 1)
        
        # 提取旋转矩阵 R_final (前三行和前三列)
        self.R = T_final[:3, :3]

        # 提取平移向量 T_final (第四列的前三行)
        self.T = T_final[:3, 3]


        # 相机内参矩阵 (P_rect_02)
        self.P_rect_02 = np.array([
            [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        ])

        # 矫正旋转矩阵 (R_rect_00)
        self.R_rect_00 = np.array([
            [9.999239e-01, 9.837760e-03, -7.445048e-03],
            [-9.869795e-03, 9.999421e-01, -4.278459e-03],
            [7.402527e-03, 4.351614e-03, 9.999631e-01]
        ])
        self.R_rect_00 = np.vstack((self.R_rect_00, [0, 0, 0]))
        self.R_rect_00 = np.hstack((self.R_rect_00, np.array([[0], [0], [0], [1]])))

        # LiDAR 到相机的变换矩阵 (Tr_velo_to_cam)
        R_velo_to_cam = np.array([
            [7.533745e-03, -9.999714e-01, -6.166020e-04],
            [1.480249e-02, 7.280733e-04, -9.998902e-01],
            [9.998621e-01, 7.523790e-03, 1.480755e-02]
        ])
        T_velo_to_cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        self.Tr_velo_to_cam = np.hstack((R_velo_to_cam, T_velo_to_cam.reshape(-1, 1)))  # 3x4 矩阵
        self.Tr_velo_to_cam = np.vstack((self.Tr_velo_to_cam, [0, 0, 0, 1]))  # 使其成为 4x4 齐次变换矩阵

        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.image_callback)
        rospy.Subscriber('/bounding_box_ros2', BoundingBox3D, self.bbox_callback)
        self._imagepub = rospy.Publisher('/plot_image', Image, queue_size=2)
        self._bboxpub = rospy.Publisher('/bbox_lidar', BoundingBoxArray, queue_size=2)

        self._current_image = None
        self.old_time = None
        print('Initialization Done')
        
    def undistort_points(self, points_2d):
        """
        应用相机畸变校正到 2D 点上。
        :param points_2d: 2D 点的坐标。
        :param K: 相机内参矩阵。
        :param D: 相机的畸变系数。
        :return: 畸变校正后的 2D 点。
        """
        points_2d_reshaped = points_2d.T.reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(points_2d_reshaped, self.K, self.D, P=self.K)
        return undistorted_points.reshape(-1, 2).T
    
    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        # print('IMG Done')
        self._current_image = image
        
    def bbox_callback(self, msg):
        """
        处理接收到的 BoundingBox3D 消息。
        """
        print(self._current_image.header.stamp.to_sec() - msg.header.stamp.to_sec())
        t1 = rospy.Time.now()
        if(self.old_time == None) :
            self.old_time = msg.header.stamp.to_sec()
        else :
            print(msg.header.stamp.to_sec() - self.old_time)
            self.old_time = msg.header.stamp.to_sec()

        cv_image = self._bridge.imgmsg_to_cv2(self._current_image, 'bgr8')
        arr_bbox = BoundingBoxArray()
        for bbox_array in msg.boxes:
            # 将 Point3DArray 转换为 NumPy 数组
            bbox_3d = np.array([[point.x, point.y, point.z] for point in bbox_array.vertices])
            result = self.project_3d_bbox_to_2d(bbox_3d, cv_image)
            
            if result is None:
                continue
            else:
                [xmin, ymin, xmax, ymax] = result
                # Create a BoundingBox message
                bbox = BoundingBox()

                # bbox.header.frame_id = msg.header.frame_id
                # bbox.header.stamp = rospy.Time.now()
                bbox.header = msg.header
                bbox.pose.position.x = xmin
                bbox.pose.position.y = ymin
                bbox.dimensions.x = xmax - xmin
                bbox.dimensions.y = ymax - ymin

                # bbox.value = scores[i]
                # bbox.label = label[i]

                arr_bbox.boxes.append(bbox)
                
                # arr_bbox.header.frame_id = msg.header.frame_id
                # arr_bbox.header.stamp = rospy.Time.now()
                # arr_bbox.header = msg.header
        arr_bbox.header = msg.header
        
        # Start Tracking in 2D
        if self._current_image is not None:
            try:
                # convert image from the subscriber into an OpenCV image
                scene = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')

                detect = []
                scores = []
                name = []
                bboxes2draw = []

                for bbox in arr_bbox.boxes:
                    detect.append([bbox.pose.position.x, bbox.pose.position.y,bbox.dimensions.x, bbox.dimensions.y])
                    scores.append(1.0)
                    name.append(np.array('person'))
                
                xywhs = torch.Tensor(detect)
                confss = torch.Tensor(scores)

                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, scene)
                for value in list(outputs):
                    x1,y1,x2,y2,track_id = value
                    bboxes2draw.append(
                        (x1, y1, x2, y2, '', track_id)
                    )


                arr_bbox = BoundingBoxArray()
                plot_bboxes(scene, bboxes2draw)
                for (x1, y1, x2, y2, cls_id, pos_id) in bboxes2draw:
                    # Create a BoundingBox message
                    bbox_msg = BoundingBox()

                    bbox_msg.header = self._current_image.header
                    bbox_msg.header.stamp = msg.header.stamp
                    bbox_msg.pose.position.x = x1
                    bbox_msg.pose.position.y = y1
                    bbox_msg.dimensions.x = x2 - x1
                    bbox_msg.dimensions.y = y2 - y1
                    bbox_msg.label = pos_id
                    arr_bbox.boxes.append(bbox_msg)
                arr_bbox.header = self._current_image.header
                arr_bbox.header.stamp = msg.header.stamp
                self._bboxpub.publish(arr_bbox)
            except CvBridgeError as e:
                print(e)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)


        # filename = f'/home/sam/Downloads/bag-synced/kitti/restore/image_{self.count}.png'  # 为每个图像生成唯一的文件名
        # self.count += 1
        # cv2.imwrite(filename, cv_image)
        self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'rgb8'))
        print('Process Time: ', rospy.Time.now() - t1)
            
    def project_3d_bbox_to_2d(self, bbox_3d, cv_image):
        """
        将 3D 边界框投影到 2D 图像平面上。
        :param K: 相机内参矩阵。
        :param D: 相机畸变系数。
        :param R: 从激光雷达到相机的旋转矩阵。
        :param T: 从激光雷达到相机的平移向量。
        :param bbox_3d: 3D 边界框的 8 个顶点。
        :return: 2D 边界框的 4 个角点 [xmin, ymin, xmax, ymax]。
        """
	
        # # 相机内参矩阵 (P_rect_02)
        # P_rect_02 = np.array([
        #     [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        #     [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        #     [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
        # ])

        # # 矫正旋转矩阵 (R_rect_00)
        # R_rect_00 = np.array([
        #     [9.999239e-01, 9.837760e-03, -7.445048e-03],
        #     [-9.869795e-03, 9.999421e-01, -4.278459e-03],
        #     [7.402527e-03, 4.351614e-03, 9.999631e-01]
        # ])
        # R_rect_00 = np.vstack((R_rect_00, [0, 0, 0]))
        # R_rect_00 = np.hstack((R_rect_00, np.array([[0], [0], [0], [1]])))

        # # LiDAR 到相机的变换矩阵 (Tr_velo_to_cam)
        # R_velo_to_cam = np.array([
        #     [7.533745e-03, -9.999714e-01, -6.166020e-04],
        #     [1.480249e-02, 7.280733e-04, -9.998902e-01],
        #     [9.998621e-01, 7.523790e-03, 1.480755e-02]
        # ])
        # T_velo_to_cam = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

        # print('Projecting 3D to 2D')
        # if self._current_image is None:
        #     return
        # # 构建变换矩阵
        # # transform = np.vstack((np.hstack((self.R, self.T[:, np.newaxis])), [0, 0, 0, 1]))

        # # 将 3D 点转换为齐次坐标
        # # ones = np.ones((bbox_3d.shape[0], 1))
        # # points_homogeneous = np.hstack((bbox_3d, ones))

        # # 应用变换
        # Tr_velo_to_cam = np.hstack((R_velo_to_cam, T_velo_to_cam.reshape(-1, 1)))  # 3x4 矩阵
        # Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))  # 使其成为 4x4 齐次变换矩阵

        # 将 bbox_3d 的每个点转换为齐次坐标
        ones = np.ones((bbox_3d.shape[0], 1))
        bbox_3d_homogeneous = np.hstack((bbox_3d, ones))

        # 应用变换
        points_camera_homogeneous = np.dot(self.P_rect_02, np.dot(self.R_rect_00, np.dot(self.Tr_velo_to_cam, bbox_3d_homogeneous.T)))

        # 保留位于相机前方的点
        front_points = points_camera_homogeneous[:, points_camera_homogeneous[2, :] > 0]

        # 检查是否还有剩余的点
        if front_points.shape[1] == 0:
            return  # 如果没有，则忽略这个边界框

        # 进行归一化以得到2D像素坐标
        front_points = front_points[:2, :] / front_points[2, :]

        # 投影到 2D
        # points_2d_homogeneous = np.dot(self.K, front_points[:3, :])
        # points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        # 应用畸变校正
        # points_2d = self.undistort_points(points_2d)

        # 找到 2D 边界框
        height, width = cv_image.shape[:2]

        xmin = max(0, min(np.min(front_points[0, :]), width - 1))
        ymin = max(0, min(np.min(front_points[1, :]), height - 1))
        xmax = max(0, min(np.max(front_points[0, :]), width - 1))
        ymax = max(0, min(np.max(front_points[1, :]), height - 1))

        if (xmin == 0 and ymin == 0 and (xmax == 0 or ymax == 0)) or (xmax == width and ymax == height and (xmin == width or ymin == height))or xmin >= xmax or ymin >= ymax:
            return None
        
        # 将边界框坐标转换为整数
        bbox_2d = np.array([xmin, ymin, xmax, ymax], dtype=int)

        #cv_image = self._bridge.imgmsg_to_cv2(self._current_image, 'bgr8')
        # 绘制边界框
        cv2.rectangle(cv_image, (bbox_2d[0], bbox_2d[1]), (bbox_2d[2], bbox_2d[3]), (0, 255, 0), 2)
        
        # filename = f'/home/sam/Downloads/bag-synced/kitti/restore/image_{self.count}.png'  # 为每个图像生成唯一的文件名
        # self.count += 1
        # cv2.imwrite(filename, cv_image)
        # self._imagepub.publish(self._bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        # return plot_image
        return [xmin, ymin, xmax, ymax]
    
    def get_bbox(self, bbox_3d):
        """
        将 3D 边界框投影到 2D 图像平面上。
        :param K: 相机内参矩阵。
        :param D: 相机畸变系数。
        :param R: 从激光雷达到相机的旋转矩阵。
        :param T: 从激光雷达到相机的平移向量。
        :param bbox_3d: 3D 边界框的 8 个顶点。
        :return: 2D 边界框的 4 个角点 [xmin, ymin, xmax, ymax]。
        """
	
        # 将 bbox_3d 的每个点转换为齐次坐标
        ones = np.ones((bbox_3d.shape[0], 1))
        bbox_3d_homogeneous = np.hstack((bbox_3d, ones))

        # 应用变换
        points_camera_homogeneous = np.dot(self.P_rect_02, np.dot(self.R_rect_00, np.dot(self.Tr_velo_to_cam, bbox_3d_homogeneous.T)))

        # 保留位于相机前方的点
        front_points = points_camera_homogeneous[:, points_camera_homogeneous[2, :] > 0]

        # 检查是否还有剩余的点
        if front_points.shape[1] == 0:
            return  # 如果没有，则忽略这个边界框

        # 进行归一化以得到2D像素坐标
        front_points = front_points[:2, :] / front_points[2, :]

        # 找到 2D 边界框
        xmin = np.min(front_points[0, :])
        ymin = np.min(front_points[1, :])
        xmax = np.max(front_points[0, :])
        ymax = np.max(front_points[1, :])

        # 将边界框坐标转换为整数
        bbox_2d = np.array([xmin, ymin, xmax, ymax], dtype=int)

        return bbox_2d
        
def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image
    
if __name__ == '__main__':

    # ROS 节点初始化
    rospy.init_node('point_cloud_to_2d', anonymous=True)
    Detector()
    rospy.spin()

    # bbox_3d = np.array(...)  # 3D 边界框的 8 个顶点

    # bbox_2d = project_3d_bbox_to_2d(K, D, R, T, bbox_3d)

