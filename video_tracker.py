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
    def image_callback(self, image):
        
        scene = self.cvb.imgmsg_to_cv2(image, 'rgb8')
        
        func_status = {}
        func_status['headpose'] = None
        result = self.det.feedCap(scene, func_status)
        result = result['face_bboxes']

        arr_bbox = BoundingBoxArray()
        for (x1, y1, x2, y2, cls_id, pos_id) in result:
            # Create a BoundingBox message
            bbox_msg = BoundingBox()

            bbox_msg.header = image.header
            bbox_msg.pose.position.x = x1
            bbox_msg.pose.position.y = y1
            bbox_msg.dimensions.x = x2 - x1
            bbox_msg.dimensions.y = y2 - y1
            bbox_msg.label = pos_id
            arr_bbox.boxes.append(bbox_msg)
        arr_bbox.header = image.header
        self.bboxpub.publish(arr_bbox)
        print('published')

def main():
    rospy.init_node('tracker_node', log_level=rospy.INFO)
    tracker = Tracker()     
    rospy.spin()

if __name__ == '__main__':
    main()