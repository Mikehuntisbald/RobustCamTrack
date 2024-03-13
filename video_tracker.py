from AIDetector_pytorch import Detector
import imutils
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
class Tracker:
    def __init__(self):
        self.bboxpub = rospy.Publisher('/bbox_camera', BoundingBoxArray, queue_size=2)
        rospy.Subscriber('/kitti/camera_color_left/image_raw', Image, self.image_callback)
        self.det = Detector()
        self.cvb = CvBridge()
        self._imagepub = rospy.Publisher('/video_tracker_img', Image, queue_size=10)

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        # Plots one bounding box on image img
        print('plot_bboxes')
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            print(x1, y1, x2, y2, cls_id, pos_id)
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
    def resize_and_pad_cv2(self, img, fill=0):
        target_width = img.shape[1]
        scale = target_width / img.shape[1]
        new_height = int(img.shape[0] * scale)
        
        # 缩放图像
        resized_img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 计算需要填充的高度
        top_pad = (target_width - new_height) // 2
        bottom_pad = target_width - new_height - top_pad
        
        # 创建一个新的图像，并填充
        # 注意cv2.copyMakeBorder的src参数是输入图像，top, bottom, left, right是填充的像素数，borderType是填充类型
        result_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[fill, fill, fill])
        
        return result_img
    def image_callback(self, image):
        
        scene = self.cvb.imgmsg_to_cv2(image, 'rgb8')
        # scene = self.resize_and_pad_cv2(scene)

        result = self.det.feedCap(scene)
        result = result['face_bboxes']
        # self.plot_bboxes(scene, result)
        self._imagepub.publish(self.cvb.cv2_to_imgmsg(scene, 'rgb8'))

        arr_bbox = BoundingBoxArray()

        # 假设frame是当前处理的帧编号，从msg1.header.stamp.to_sec()获取
        frame = int(image.header.stamp.to_sec())

        # 示例文件名，您可以根据需要修改
        output_filename = "/home/sam/tracking_output.txt"
        # for (x1, y1, x2, y2, cls_id, pos_id) in result:
        #     print(x1, y1, x2, y2, cls_id, pos_id)
        with open(output_filename, "a") as file:
            for (x1, y1, x2, y2, cls_id, pos_id) in result:
                # Create a BoundingBox message
                bbox_msg = BoundingBox()

                bbox_msg.header = image.header
                bbox_msg.pose.position.x = x1
                bbox_msg.pose.position.y = y1
                bbox_msg.dimensions.x = x2 - x1
                bbox_msg.dimensions.y = y2 - y1
                bbox_msg.label = pos_id
                bbox_msg.header.frame_id = cls_id
                # 构造KITTI格式的字符串
                kitti_format_str = f"{frame} {pos_id} Car -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10\n"

                # 写入文件
                file.write(kitti_format_str)

                arr_bbox.boxes.append(bbox_msg)
        arr_bbox.header = image.header
        self.bboxpub.publish(arr_bbox)
        print('image.header stamp: ', image.header.stamp)

def main():
    rospy.init_node('tracker_node', log_level=rospy.INFO)
    tracker = Tracker()     
    rospy.spin()

if __name__ == '__main__':
    main()
