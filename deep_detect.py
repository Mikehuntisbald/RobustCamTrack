#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
import pandas
import cv2
import matplotlib.pyplot as plt 
import torch

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort.sort.detection import Detection
from deep_sort.deep_sort.sort.tracker import Tracker
from deep_sort.deep_sort.deep_sort_lidar import DeepSort
from deep_sort.deep_sort.sort import preprocessing as prep

from tracker.msg import TrackerObject, TrackerObjectArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray



class Detector: 

   def __init__(self):
      self.deep_weights = '/home/sam/iv_ws/src/tracker/src/deep_sort/model_data/mars-small128.pb'
      self.yolo_weights = '/home/sam/adversarial-robustness-toolbox/notebooks/adversarial_patch/yolov5n.pt'
      self.yolov5 = '/home/sam/yolov5'
      image_topic = rospy.get_param('~image_topic')
      point_cloud_topic = rospy.get_param('~point_cloud_topic', None)

      self._global_frame = None #'camera'
      self._frame = 'camera_depth_frame'
      self._tf_prefix = rospy.get_param('~tf_prefix', rospy.get_name())
      #################################################
      self.model = torch.hub.load(self.yolov5, 'custom', path=self.yolo_weights, source='local')
    #   self.model.classes = [0]
      self.model.eval()
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.model.to(device)

      self.id = 0
      # create a transform listener so we get the fixed frame the user wants
      # to publish object tfs related to
      self._tf_listener = tf.TransformListener()
      self.count = 0

      # create detector
      self._bridge = CvBridge()

      # image and point cloud subscribers
      # and variables that will hold their values
      rospy.Subscriber(image_topic, Image, self.image_callback)

      if point_cloud_topic is not None:
         rospy.Subscriber(point_cloud_topic, PointCloud2, self.pc_callback, queue_size=20)
      else:
         rospy.loginfo(
            'No point cloud information available. Objects will not be placed in the scene.')

      self._current_image = None
      self._current_pc = None

      # publisher for frames with detected objects
      self._imagepub = rospy.Publisher('~labeled_detect', Image, queue_size=10)

      self._tfpub = tf.TransformBroadcaster()
      rospy.loginfo('Ready to detect!')
      self._bboxpub = rospy.Publisher('/bbox_camera', BoundingBoxArray, queue_size=2)

   def image_callback(self, image):
      """Image callback"""
      # Store value on a private attribute
      self._current_image = image
      # only run if there's an image present
      if self._current_image is not None:
         try:

               # if the user passes a fixed frame, we'll ask for transformation
               # vectors from the camera link to the fixed frame
               if self._global_frame is not None:
                  (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame,
                                                                  '/'+ self._frame,
                                                                  rospy.Time(0))

               # convert image from the subscriber into an OpenCV image
               scene = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')
               
               # Inference
               results = self.model(scene, size=640)
               # print(results)
               # results.xyxy[0] 
               deteccao = results.pandas().xyxy[0]
               # marked = results.render()
               # marked_image = np.squeeze(marked)
               # [scene] = self.apply_patch_to_images([scene], deteccao, '/home/sam/Downloads/bag-synced/kitti/patch.png')
               # results = self.model(scene, size=640)
               # deteccao = results.pandas().xyxy[0]

               detect = []
               scores = []
               name = []
               for i in range(len(deteccao)):
                  detect.append(np.array([deteccao['xmin'][i], deteccao['ymin'][i], deteccao['xmax'][i] - deteccao['xmin'][i], deteccao['ymax'][i] - deteccao['ymin'][i]]))
                  scores.append(np.array(deteccao['confidence'][i]))
                  name.append(np.array(deteccao['name'][i]))
               # print(name)
               metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
               tracker = Tracker(metric, max_iou_distance = 1.7, max_age = 100, n_init = 5)
               model_filename = self.deep_weights #Change it to your directory
               encoder = generate_detections.create_box_encoder(model_filename)

               features = encoder(scene, detect)
            #  self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'bgr8'))
               detections_new = [Detection(bbox, score, feature) for bbox, score, feature in zip(detect,scores, features)] # cria um objeto detection para cada deteccao
               # Run non-maxima suppression.
               boxes = np.array([d.tlwh for d in detections_new]) # cria um array com as coordenadas de cada deteccao
               scores_new = np.array([d.confidence for d in detections_new]) # cria um array com as confiancas de cada deteccao
               indices = prep.non_max_suppression(boxes, 1.0 , scores_new) # aplica o non-maxima suppression para eliminar deteccoes duplicadas
               detections_new = [detections_new[i] for i in indices] # cria um array com as deteccoes que sobraram
               name_new = [name[i] for i in indices]
               scores_new = [scores_new[i] for i in indices]

               # Call the tracker
               tracker.predict() # faz a predicao do tracker
               tracker.update(detections_new) # atualiza o estado do objeto
               #lidar camera同时检测跟踪，lidar的纹理使用标定后相机对应的纹理
               #每次得到结果后，使用匹配算法，将lidar和camera的结果进行匹配，并维护一个哈希表，只有当两个结果都有时才进行发布，有一个有的话就修正另一个的box
               publishers = {}
               # cv_image = self._bridge.imgmsg_to_cv2(self._current_image, 'bgr8')
               print(len(tracker.tracks))
               # [cv_image] = self.apply_patch_to_images([cv_image], deteccao, '/home/sam/Downloads/bag-synced/kitti/patch.png')
               # filename = f'/home/sam/Downloads/bag-synced/kitti/patched/image_{self.count}.png'  # 为每个图像生成唯一的文件名
               # self.count += 1
               # cv2.imwrite(filename, cv_image)
               arr_bbox = BoundingBoxArray()

               for track, class_name, score in zip(tracker.tracks, name_new, scores_new):

                  if score < 0.3 or class_name != 'person':
                        continue
                  if track.is_confirmed() and track.time_since_update > 10: # se o objeto foi confirmado e nao foi atualizado a mais de 1 frame
                     continue
                  # print(track.is_tentative()) # 
                  bbox = track.to_tlbr()
                  track_id = track.track_id

                  xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

                   # Create a BoundingBox message
                  bbox_msg = BoundingBox()

                  # bbox.header.frame_id = msg.header.frame_id
                  # bbox.header.stamp = rospy.Time.now()
                  bbox_msg.header = self._current_image.header
                  bbox_msg.pose.position.x = xmin
                  bbox_msg.pose.position.y = ymin
                  bbox_msg.dimensions.x = xmax - xmin
                  bbox_msg.dimensions.y = ymax - ymin
                  bbox_msg.label = track_id
                  # bbox_msg.value = scores[i]
                  # bbox_msg.label = label[i]

                  arr_bbox.boxes.append(bbox_msg)

                  cv2.rectangle(scene, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                  # 绘制类别、ID 和 分数
                  # label = f"{class_name} ID: {track_id} Score: {score:.2f}"
                  label = f"ID: {track_id}"
                  cv2.putText(scene, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4)

                  id = track.track_id
                  publishers[id] = rospy.Publisher('~person_' + str(id), TrackerObjectArray, queue_size=10)
   
                  publish_tf = False
               
               arr_bbox.header = self._current_image.header
               self._bboxpub.publish(arr_bbox)
               # filename = f'/home/sam/Downloads/bag-synced/kitti/ori/image_{self.count}.png'  # 为每个图像生成唯一的文件名
               # self.count += 1
               # cv2.imwrite(filename, scene)
               self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'bgr8'))

         except CvBridgeError as e:
               print(e)
         except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
               print(e)
   def pc_callback(self, pc):
      """Point cloud callback"""
      # Store value on a private attribute
      self._current_pc = pc

   def apply_patch_to_images(self, x, deteccao, patch_filename):
      # 读取补丁图像
      patch = cv2.imread(patch_filename)
      if patch is None:
         raise ValueError(f"Could not load patch image from {patch_filename}")

      # 遍历每张图像和其对应的检测结果
      for i, image in enumerate(x):
         for j in range(len(deteccao)):
               # 检查是否为车辆
               # if deteccao['class'][j] == 'car':
                  # 计算边界框的中心和尺寸
               xmin, ymin, xmax, ymax = deteccao['xmin'][j], deteccao['ymin'][j], deteccao['xmax'][j], deteccao['ymax'][j]
               center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
               box_width, box_height = xmax - xmin, ymax - ymin

               # 根据边界框尺寸调整补丁大小
               patch_size = int(4 * min(box_width, box_height) / 9)
               resized_patch = cv2.resize(patch, (patch_size, patch_size))

               # 计算补丁的位置
               patch_x1 = max(int(center_x - patch_size // 2), 0)
               patch_y1 = max(int(center_y - patch_size // 2), 0)
               patch_x2 = min(int(patch_x1 + patch_size), image.shape[1])
               patch_y2 = min(int(patch_y1 + patch_size), image.shape[0])

               # 将补丁贴到图像上
               image[patch_y1:patch_y2, patch_x1:patch_x2] = resized_patch[:patch_y2-patch_y1, :patch_x2-patch_x1]

      return x

   def run(self):
      # run while ROS runs
      while not rospy.is_shutdown():
         # only run if there's an image present
         if self._current_image is not None:
            try:

                # if the user passes a fixed frame, we'll ask for transformation
                # vectors from the camera link to the fixed frame
                if self._global_frame is not None:
                  (trans, _) = self._tf_listener.lookupTransform('/' + self._global_frame,
                                                                 '/'+ self._frame,
                                                                 rospy.Time(0))

                # convert image from the subscriber into an OpenCV image
                scene = self._bridge.imgmsg_to_cv2(self._current_image, 'rgb8')
                
                # Inference
                results = self.model(scene, size=640)
                # print(results)
                # results.xyxy[0] 
                deteccao = results.pandas().xyxy[0]
                # marked = results.render()
                # marked_image = np.squeeze(marked)

                detect = []
                scores = []
                name = []
                for i in range(len(deteccao)):
                    detect.append(np.array([deteccao['xmin'][i], deteccao['ymin'][i], deteccao['xmax'][i] - deteccao['xmin'][i], deteccao['ymax'][i] - deteccao['ymin'][i]]))
                    scores.append(np.array(deteccao['confidence'][i]))
                    name.append(np.array(deteccao['name'][i]))
                # print(name)
                metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)
                tracker = Tracker(metric, max_iou_distance = 0.7, max_age = 70, n_init = 3)
                model_filename = self.deep_weights #Change it to your directory
                encoder = generate_detections.create_box_encoder(model_filename)

                features = encoder(scene, detect)
               #  self._imagepub.publish(self._bridge.cv2_to_imgmsg(scene, 'bgr8'))
                detections_new = [Detection(bbox, score, feature) for bbox,score, feature in zip(detect,scores, features)] # cria um objeto detection para cada deteccao
                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections_new]) # cria um array com as coordenadas de cada deteccao
                scores_new = np.array([d.confidence for d in detections_new]) # cria um array com as confiancas de cada deteccao
                indices = prep.non_max_suppression(boxes, 1.0 , scores_new) # aplica o non-maxima suppression para eliminar deteccoes duplicadas
                detections_new = [detections_new[i] for i in indices] # cria um array com as deteccoes que sobraram
                name_new = [name[i] for i in indices]
                scores_new = [scores_new[i] for i in indices]

                # Call the tracker
                tracker.predict() # faz a predicao do tracker
                tracker.update(detections_new) # atualiza o estado do objeto
                #lidar camera同时检测跟踪，lidar的纹理使用标定后相机对应的纹理
                #每次得到结果后，使用匹配算法，将lidar和camera的结果进行匹配，并维护一个哈希表，只有当两个结果都有时才进行发布，有一个有的话就修正另一个的box
                publishers = {}
                cv_image = self._bridge.imgmsg_to_cv2(self._current_image, 'bgr8')
                print(len(tracker.tracks))
               #  [cv_image] = self.apply_patch_to_images([cv_image], deteccao, '/home/sam/Downloads/bag-synced/kitti/patch.png')
               #  filename = f'/home/sam/Downloads/bag-synced/kitti/patched/image_{self.count}.png'  # 为每个图像生成唯一的文件名
               #  self.count += 1
               #  cv2.imwrite(filename, cv_image)
                for track, class_name, score in zip(tracker.tracks, name_new, scores_new):
                    if score < 0.8:
                        continue
                    if track.is_confirmed() and track.time_since_update > 1: # se o objeto foi confirmado e nao foi atualizado a mais de 1 frame
                        continue
                    # print(track.is_tentative()) # 
                    bbox = track.to_tlbr()
                    track_id = track.track_id

                    ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
                  #   cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    # 绘制类别、ID 和 分数
                    label = f"{class_name} ID: {track_id} Score: {score:.2f}"
                  #   cv2.putText(cv_image, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)

                    id = track.track_id
                    publishers[id] = rospy.Publisher('~person_' + str(id), TrackerObjectArray, queue_size=10)
    
                    publish_tf = False
                    # if self._current_pc is None:
                    #     rospy.loginfo(
                    #         'No point cloud information available to track current object in scene')

                    # # if there is point cloud data, we'll try to place a tf
                    # # in the object's location
                    # else:
                    #     y_center = round(ymax - ((ymax - ymin) / 2))
                    #     x_center = round(xmax - ((xmax - xmin) / 2))
                    #     # this function gives us a generator of points.
                    #     # we ask for a single point in the center of our object.
                    #     pc_list = list(
                    #         pc2.read_points(self._current_pc,
                    #                     skip_nans=True,
                    #                     field_names=('x', 'y', 'z'),
                    #                     uvs=[(x_center, y_center)]))

                    #     if len(pc_list) > 0:
                    #         publish_tf = True
                    #         # this is the location of our object in space
                    #         tf_id = 'Person' + '_' + str(id)

                    #         # if the user passes a tf prefix, we append it to the object tf name here
                    #         if self._tf_prefix is not None:
                    #             tf_id = self._tf_prefix + '/' + tf_id

                    #         tf_id = tf_id

                    #         point_x, point_y, point_z = pc_list[0]

                    # # we'll publish a TF related to this object only once
                    # if publish_tf:
                    #     # kinect here is mapped as camera_link
                    #     # object tf (x, y, z) must be
                    #     # passed as (z,-x,-y)
                    #     object_tf = [point_z, -point_x, -point_y]
                    #     frame = self._frame

                    #     # translate the tf in regard to the fixed frame
                    #     if self._global_frame is not None:
                    #         object_tf = np.array(trans) + object_tf
                    #         frame = self._global_frame

                    #     # this fixes #7 on GitHub, when applying the
                    #     # translation to the tf creates a vector that
                    #     # RViz just can'y handle
                    #     if object_tf is not None:
                    #         self._tfpub.sendTransform((object_tf),
                    #                                 tf.transformations.quaternion_from_euler(
                    #                                 0, 0, 0),
                    #                                 rospy.Time.now(),
                    #                                 tf_id,
                    #                                 frame)
                self._imagepub.publish(self._bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

            except CvBridgeError as e:
                print(e)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print(e)


if __name__ == '__main__':
   rospy.init_node('tracker_node', log_level=rospy.INFO)
   detector = Detector()
   rospy.spin()
   # try:
   #    Detector().run()
   # except KeyboardInterrupt:
   #    rospy.loginfo('Shutting down')
