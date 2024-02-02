// Copyright 2021 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lidar_centerpoint/node.hpp"

#include <lidar_centerpoint/centerpoint_config.hpp>
#include <lidar_centerpoint/preprocess/pointcloud_densification.hpp>
#include <lidar_centerpoint/ros_utils.hpp>
#include <lidar_centerpoint/utils.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include "object_recognition_utils/object_recognition_utils.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"
#include "tier4_autoware_utils/math/constants.hpp"

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{

struct Point3D {
    double x, y, z;
};

LidarCenterPointNode::LidarCenterPointNode(const rclcpp::NodeOptions & node_options)
: Node("lidar_center_point", node_options), tf_buffer_(this->get_clock())
{
  const float score_threshold =
    static_cast<float>(this->declare_parameter<double>("score_threshold", 0.35));
  const float circle_nms_dist_threshold =
    static_cast<float>(this->declare_parameter<double>("circle_nms_dist_threshold"));
  const auto yaw_norm_thresholds =
    this->declare_parameter<std::vector<double>>("yaw_norm_thresholds");
  const std::string densification_world_frame_id =
    this->declare_parameter("densification_world_frame_id", "map");
  const int densification_num_past_frames =
    this->declare_parameter("densification_num_past_frames", 1);
  const std::string trt_precision = this->declare_parameter("trt_precision", "fp16");
  const std::string encoder_onnx_path = this->declare_parameter<std::string>("encoder_onnx_path");
  const std::string encoder_engine_path =
    this->declare_parameter<std::string>("encoder_engine_path");
  const std::string head_onnx_path = this->declare_parameter<std::string>("head_onnx_path");
  const std::string head_engine_path = this->declare_parameter<std::string>("head_engine_path");
  class_names_ = this->declare_parameter<std::vector<std::string>>("class_names");
  has_twist_ = this->declare_parameter("has_twist", false);
  const std::size_t point_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("point_feature_size"));
  const std::size_t max_voxel_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("max_voxel_size"));
  const auto point_cloud_range = this->declare_parameter<std::vector<double>>("point_cloud_range");
  const auto voxel_size = this->declare_parameter<std::vector<double>>("voxel_size");
  const std::size_t downsample_factor =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("downsample_factor"));
  const std::size_t encoder_in_feature_size =
    static_cast<std::size_t>(this->declare_parameter<std::int64_t>("encoder_in_feature_size"));
  const auto allow_remapping_by_area_matrix =
    this->declare_parameter<std::vector<int64_t>>("allow_remapping_by_area_matrix");
  const auto min_area_matrix = this->declare_parameter<std::vector<double>>("min_area_matrix");
  const auto max_area_matrix = this->declare_parameter<std::vector<double>>("max_area_matrix");

  detection_class_remapper_.setParameters(
    allow_remapping_by_area_matrix, min_area_matrix, max_area_matrix);

  {
    NMSParams p;
    p.nms_type_ = NMS_TYPE::IoU_BEV;
    p.target_class_names_ =
      this->declare_parameter<std::vector<std::string>>("iou_nms_target_class_names");
    p.search_distance_2d_ = this->declare_parameter<double>("iou_nms_search_distance_2d");
    p.iou_threshold_ = this->declare_parameter<double>("iou_nms_threshold");
    iou_bev_nms_.setParameters(p);
  }

  NetworkParam encoder_param(encoder_onnx_path, encoder_engine_path, trt_precision);
  NetworkParam head_param(head_onnx_path, head_engine_path, trt_precision);
  DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);

  if (point_cloud_range.size() != 6) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_centerpoint"),
      "The size of point_cloud_range != 6: use the default parameters.");
  }
  if (voxel_size.size() != 3) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_centerpoint"),
      "The size of voxel_size != 3: use the default parameters.");
  }
  CenterPointConfig config(
    class_names_.size(), point_feature_size, max_voxel_size, point_cloud_range, voxel_size,
    downsample_factor, encoder_in_feature_size, score_threshold, circle_nms_dist_threshold,
    yaw_norm_thresholds);
  detector_ptr_ =
    std::make_unique<CenterPointTRT>(encoder_param, head_param, densification_param, config);
  rclcpp::QoS qos_profile(100); // 设置 depth 为 10
        qos_profile.reliable(); // 设置为可靠传输
  pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "~/input/pointcloud", qos_profile,
    std::bind(&LidarCenterPointNode::pointCloudCallback, this, std::placeholders::_1));
    // rclcpp::SensorDataQoS{}.keep_last(100)
  objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>(
    "~/output/objects", rclcpp::QoS{10});
  box_pub_ = this->create_publisher<bounding_box_3d::msg::BoundingBox3D>(
    "/bounding_box_ros2", rclcpp::QoS{10});
  // initialize debug tool
  {
    using tier4_autoware_utils::DebugPublisher;
    using tier4_autoware_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, "lidar_centerpoint");
    stop_watch_ptr_->tic("cyclic_time");
    stop_watch_ptr_->tic("processing_time");
  }

  if (this->declare_parameter("build_only", false)) {
    RCLCPP_INFO(this->get_logger(), "TensorRT engine is built and shutdown node.");
    rclcpp::shutdown();
  }
}

void LidarCenterPointNode::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_pointcloud_msg)
{
  auto stamp = input_pointcloud_msg->header.stamp;

  // ROS2中的时间戳是以秒和纳秒为单位
  auto seconds = stamp.sec;
  auto nanoseconds = stamp.nanosec;

  // 将纳秒转换为小数秒
  double fractional_seconds = static_cast<double>(nanoseconds) / 1e9;

  // 合并秒和小数秒
  double timestamp = seconds + fractional_seconds;

  // 打印时间戳，格式为 "秒.小数秒"
  std::cout << "Timestamp: " << std::fixed << std::setprecision(9) << timestamp << " seconds" << std::endl;

  using Label = autoware_auto_perception_msgs::msg::ObjectClassification;
  const auto objects_sub_count =
    objects_pub_->get_subscription_count() + objects_pub_->get_intra_process_subscription_count();
  if (objects_sub_count < 1) {
    return;
  }

  if (stop_watch_ptr_) {
    stop_watch_ptr_->toc("processing_time", true);
  }

  std::vector<Box3D> det_boxes3d;
  bool is_success = detector_ptr_->detect(*input_pointcloud_msg, tf_buffer_, det_boxes3d);
  if (!is_success) {
    return;
  }

  std::vector<autoware_auto_perception_msgs::msg::DetectedObject> raw_objects;
  raw_objects.reserve(det_boxes3d.size());
  for (const auto & box3d : det_boxes3d) {
    autoware_auto_perception_msgs::msg::DetectedObject obj;
    box3DToDetectedObject(box3d, class_names_, has_twist_, obj);
    // if(obj.classification[0].label == Label::BICYCLE) 
    raw_objects.emplace_back(obj);
  }

  autoware_auto_perception_msgs::msg::DetectedObjects output_msg;
  output_msg.header = input_pointcloud_msg->header;
  output_msg.objects = iou_bev_nms_.apply(raw_objects);

  bounding_box_3d::msg::BoundingBox3D msg;
  for(const auto& object : raw_objects){
    if(object.classification[0].label != Label::BICYCLE){
      continue;
    }
    // std::cout<<"car"<<std::endl;
    bounding_box_3d::msg::Point3DArray vertices;
    vertices.vertices.resize(8);
    double siny_cosp = 2 * (object.kinematics.pose_with_covariance.pose.orientation.w * object.kinematics.pose_with_covariance.pose.orientation.z + object.kinematics.pose_with_covariance.pose.orientation.x * object.kinematics.pose_with_covariance.pose.orientation.y);
    double cosy_cosp = 1 - 2 * (object.kinematics.pose_with_covariance.pose.orientation.y * object.kinematics.pose_with_covariance.pose.orientation.y + object.kinematics.pose_with_covariance.pose.orientation.z * object.kinematics.pose_with_covariance.pose.orientation.z);
    double yaw = atan2(siny_cosp, cosy_cosp);

    // Rotate point back by -yaw around z-axis
    float cos_yaw = cos(yaw);
    float sin_yaw = sin(yaw);

      // 边界框的半长、半宽和半高
    double half_length = object.shape.dimensions.x / 2.0;
    double half_width = object.shape.dimensions.y / 2.0;
    double half_height = object.shape.dimensions.z / 2.0;

    // 计算边界框顶点相对于中心点的局部坐标
    std::vector<Point3D> local_vertices = {
        { half_length,  half_width,  half_height},
        { half_length, -half_width,  half_height},
        {-half_length, -half_width,  half_height},
        {-half_length,  half_width,  half_height},
        { half_length,  half_width, -half_height},
        { half_length, -half_width, -half_height},
        {-half_length, -half_width, -half_height},
        {-half_length,  half_width, -half_height}
    };

    // 应用旋转和平移变换
    for (int i = 0; i < 8; ++i) {
        vertices.vertices[i].x = object.kinematics.pose_with_covariance.pose.position.x + cos_yaw * local_vertices[i].x - sin_yaw * local_vertices[i].y;
        vertices.vertices[i].y = object.kinematics.pose_with_covariance.pose.position.y + sin_yaw * local_vertices[i].x + cos_yaw * local_vertices[i].y;
        vertices.vertices[i].z = object.kinematics.pose_with_covariance.pose.position.z + local_vertices[i].z - 0.25;
    }

    msg.boxes.push_back(vertices);
    msg.header = input_pointcloud_msg->header;
  }

  box_pub_->publish(msg);

  detection_class_remapper_.mapClasses(output_msg);

  if (objects_sub_count > 0) {
    objects_pub_->publish(output_msg);
  }
  const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
  std::cout<<"processing_time_ms: "<<processing_time_ms<<std::endl;
  // add processing time for debug
  if (debug_publisher_ptr_ && stop_watch_ptr_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic_time", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing_time", true);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time_ms", processing_time_ms);
  }
}

}  // namespace centerpoint

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(centerpoint::LidarCenterPointNode)