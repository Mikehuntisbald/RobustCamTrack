cmake_minimum_required(VERSION 3.14)
project(bounding_box_3d)

# Find required dependencies
find_package(ament_cmake REQUIRED)
#find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# Generate ROS interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Point3DArray.msg"
  "msg/Point3D.msg"
  "msg/BoundingBox3D.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)
install(FILES mapping_rules.yaml
  DESTINATION share/${PROJECT_NAME}
)
ament_export_dependencies(rosidl_default_runtime)

# This macro ensures modules and global scripts declared therein get installed
# See: http://wiki.ros.org/catkin/CMakeLists.txt#Installing
ament_package()


