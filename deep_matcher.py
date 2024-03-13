#!/usr/bin/env python
import message_filters
import rospy
import numpy as np
import tf
import pandas
import cv2
import matplotlib.pyplot as plt 
import torch
import random

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2

from deep_sort.deep_sort.sort.detection import Detection
from deep_sort.deep_sort.sort import nn_matching
from deep_sort.deep_sort.sort.matcher import Matcher
# from deep_sort import generate_detections
from deep_sort.deep_sort.sort import preprocessing as prep

# from tracker.msg import TrackerObject, TrackerObjectArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

class Correlator:
    def __init__(self):
        image_topic = '/kitti/camera_color_left/image_raw' #rospy.get_param('~image_topic')
        # 创建两个订阅者，订阅不同的主题
        sub1 = message_filters.Subscriber('/bbox_camera', BoundingBoxArray)
        sub2 = message_filters.Subscriber('/bbox_lidar', BoundingBoxArray)
        sub3 = message_filters.Subscriber(image_topic, Image)
        # rospy.Subscriber(image_topic, Image, self.image_callback)
        
        # 使用 ApproximateTime 策略进行软同步
        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2, sub3], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)
        
        self._bridge = CvBridge()
        # self.deep_weights = '/home/sam/iv_ws/src/tracker/src/deep_sort/model_data/mars-small128.pb'
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
        self.matcher = Matcher(self.metric, max_iou_distance = 0.7, max_age = 70, n_init = 3)
        self._imagepub = rospy.Publisher('/labeled_detect', Image, queue_size=10)

        self.count = 0
        self.selected_track_id = 0
        self.save = []
        print('init')

    def callback(self, msg1, msg2, msg3):
        flag = rospy.get_param('/flag', False)
        print("frame", int(msg1.header.stamp.to_sec()))
        # 把msg转换成能处理的Detection格式
        # model_filename = self.deep_weights #Change it to your directory
        # encoder = generate_detections.create_box_encoder(model_filename)
        scene = self._bridge.imgmsg_to_cv2(msg3, 'rgb8')
        detect1 = []
        scores1 = []
        name1 = []
        id1 = []
        for bbox in msg1.boxes:
            detect1.append(np.array([bbox.pose.position.x, bbox.pose.position.y, bbox.dimensions.x, bbox.dimensions.y]))
            scores1.append(np.array(1.0))
            name1.append(np.array('car'))
            id1.append(np.array(bbox.label))
        # features1 = encoder(scene, detect1)
        features1 = None
        # detections_1 = [Detection(bbox, score, feature) for bbox,score, feature in zip(detect1,scores1, features1)]
        detections_1 = [Detection(bbox, 'car', score, None) for bbox,score in zip(detect1,scores1)]

        detect2 = []
        scores2 = []
        name2 = []
        id2 = []
        for bbox in msg2.boxes:
            detect2.append(np.array([bbox.pose.position.x, bbox.pose.position.y, bbox.dimensions.x, bbox.dimensions.y]))
            scores2.append(np.array(1.0))
            name2.append(np.array('car'))
            id2.append(np.array(bbox.label))
        # features2 = encoder(scene, detect2)
        features2 = None
        # detections_2 = [Detection(bbox, score, feature) for bbox,score, feature in zip(detect2,scores2, features2)]
        detections_2 = [Detection(bbox, 'car', score, None) for bbox,score in zip(detect2,scores2)]

        for ind, detection in enumerate(detections_1):
            self.matcher._initiate_track_1(detection, id1[ind])
        for ind, detection in enumerate(detections_2):
            self.matcher._initiate_track_2(detection, id2[ind])

        # if flag == True:
        #     if (self.matcher.tracks_1 and self.count >= 10) or self.selected_track_id == 0:
        #         # 随机选择一个元素
        #         self.selected_track_id = random.choice(self.matcher.tracks_1).track_id.item()
        #         if self.selected_track_id not in self.save:
        #             pass
        #         else:
        #             wbbox = [0, 0, 0, 0]
        #             for track in self.matcher.tracks_1 :
        #                 if track.track_id.item() == self.selected_track_id :
        #                     wbbox = track.to_tlbr()
        #                     break
        #             # 保留未被选中的元素
        #             self.matcher.tracks_1 = [track for track in self.matcher.tracks_1 if track.track_id.item() != self.selected_track_id]

        #             cv2.rectangle(scene, (int(wbbox[0]), int(wbbox[1])), (int(wbbox[2]), int(wbbox[3])), (255, 255, 255), -1)
        #             self.count = 1
        #     else:
        #         wbbox = [0, 0, 0, 0]
        #         for track in self.matcher.tracks_1 :
        #             if track.track_id.item() == self.selected_track_id:
        #                 wbbox = track.to_tlbr()
        #                 break
        #         self.matcher.tracks_1 = [track for track in self.matcher.tracks_1 if track.track_id.item() != self.selected_track_id]
        #         cv2.rectangle(scene, (int(wbbox[0]), int(wbbox[1])), (int(wbbox[2]), int(wbbox[3])), (255, 255, 255), -1)
        #         self.count+=1
            
        wbbox = [0, 0, 0, 0]
        if flag == True:
            if (self.matcher.tracks_1 and self.count >= 10) or self.selected_track_id == 0:
                # 确定需要保留的轨迹数量（一半）
                num_tracks_to_keep = len(self.matcher.tracks_1) // 2
                # 随机选择一半的轨迹
                selected_tracks = random.sample(self.matcher.tracks_1, num_tracks_to_keep)
                
                # 对于每个被选中的轨迹
                for track in selected_tracks:
                    # 获取轨迹的边界框并绘制
                    wbbox = track.to_tlbr()
                    cv2.rectangle(scene, (int(wbbox[0]), int(wbbox[1])), (int(wbbox[2]), int(wbbox[3])), (255, 255, 255), -1)
                    # 保留未被选中的元素
                    self.matcher.tracks_1 = [track for track in self.matcher.tracks_1 if track not in selected_tracks]

                    # cv2.rectangle(scene, (int(wbbox[0]), int(wbbox[1])), (int(wbbox[2]), int(wbbox[3])), (255, 255, 255), -1)
                    self.count = 1
            else:
                
                selected_tracks = random.sample(self.matcher.tracks_1, num_tracks_to_keep)
        
                for track in selected_tracks:
                    wbbox = track.to_tlbr()
                    cv2.rectangle(scene, (int(wbbox[0]), int(wbbox[1])), (int(wbbox[2]), int(wbbox[3])), (255, 255, 255), -1)
                
                self.matcher.tracks_1 = [track for track in self.matcher.tracks_1 if track not in selected_tracks]
                self.count+=1

        self.matcher.update(detections_2)

        # 假设frame是当前处理的帧编号，从msg1.header.stamp.to_sec()获取
        frame = int(msg1.header.stamp.to_sec())

        # 示例文件名，您可以根据需要修改
        output_filename = "/home/sam/tracking_output.txt"
        # 打开文件准备写入
        # print(self.save)
        with open(output_filename, "a") as file:
            for track in self.matcher.tracks_1:
                ## 第一次出现，且没有匹配掠过
                # if track.track_id.item() not in self.save and self.matcher.current_track_id_map1.get(track.track_id.item(), None) is None:
                #     print('omitted')
                #     continue

                bbox = track.to_tlbr()
                track_id = track.track_id

                # 构造KITTI格式的字符串
                kitti_format_str = f"{frame} {track_id} Car -1 -1 -10 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} -1 -1 -1 -1000 -1000 -1000 -10\n"

                # 写入文件
                # file.write(kitti_format_str)

                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                # cv2.rectangle(scene, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                # 绘制类别、ID 和 分数
                # label = f"{class_name} ID: {track_id} Score: {score:.2f}"
                label = f"{track_id}"

                font_scale = 1.0  # 字体大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2  # 文本线条厚度
                # if track.restored and flag == True:
                # if track.track_id.item() not in self.save and self.matcher.current_track_id_map1.get(track.track_id.item(), None) is None:
                #     print('omitted')
                #     cv2.putText(scene, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                #     cv2.rectangle(scene, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                #     continue
                if track.restored and flag == False:
                    # label = f"ID: {track_id} Retrieved"
                    # cv2.putText(scene, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 3)
                    # cv2.rectangle(scene, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), -1)

                    cv2.putText(scene, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.rectangle(scene, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
                    
                    file.write(kitti_format_str)
                    text = "Attack Detected"
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = scene.shape[1] - text_size[0] - 10  # 距离右边 10 像素
                    text_y = text_size[1] + 10  # 距离顶部 10 像素
                    # 设置字体颜色为红色
                    font_color = (255, 0, 0)
                    # 在图像上绘制文本
                    # cv2.putText(scene, text, (text_x, text_y), font, font_scale, font_color, thickness)
                else:
                    cv2.putText(scene, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.rectangle(scene, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    file.write(kitti_format_str)
                    # text = "No Attack"
                    # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    # text_x = scene.shape[1] - text_size[0] - 10  # 距离右边 10 像素
                    # text_y = text_size[1] + 10  # 距离顶部 10 像素
                    # # 设置字体颜色为绿色
                    # font_color = (0, 255, 0)
                    # # 在图像上绘制文本
                    # cv2.putText(scene, text, (text_x, text_y), font, font_scale, font_color, thickness)
        self.save = [track.track_id.item() for track in self.matcher.tracks_1]
        self.matcher.tracks_1 = []
        self.matcher.tracks_2 = []
        self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'rgb8'))
        
    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self._current_image = image

    def listener(self):
        image_topic = rospy.get_param('~image_topic')
        # 创建两个订阅者，订阅不同的主题
        sub1 = message_filters.Subscriber('/bbox_camera', BoundingBoxArray)
        sub2 = message_filters.Subscriber('/bbox_lidar', BoundingBoxArray)
        rospy.Subscriber(image_topic, Image, self.image_callback)
        
        # 使用 ApproximateTime 策略进行软同步
        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)


if __name__ == '__main__':
    rospy.init_node('jsk_msgs_sync_subscriber', anonymous=True)
    correlator = Correlator()
    # correlator.listener()
    rospy.spin()