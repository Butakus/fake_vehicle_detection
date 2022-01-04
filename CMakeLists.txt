cmake_minimum_required(VERSION 3.5)
project(fake_vehicle_detection)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(loco_framework REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Platoon distance estimator target
add_library(platoon_distance_estimator_node SHARED
  src/platoon_distance_estimator.cpp
)
target_include_directories(platoon_distance_estimator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(platoon_distance_estimator_node
  rclcpp
  rclcpp_components
  geometry_msgs
  loco_framework
)
rclcpp_components_register_node(platoon_distance_estimator_node
  PLUGIN "fake_vehicle_detection::PlatoonDistanceEstimator"
  EXECUTABLE platoon_distance_estimator
)

# Platoon multi-distance estimator target
add_library(platoon_multi_distance_estimator_node SHARED
  src/platoon_multi_distance_estimator.cpp
)
target_include_directories(platoon_multi_distance_estimator_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(platoon_multi_distance_estimator_node
  rclcpp
  rclcpp_components
  geometry_msgs
  loco_framework
)
rclcpp_components_register_node(platoon_multi_distance_estimator_node
  PLUGIN "fake_vehicle_detection::PlatoonMultiDistanceEstimator"
  EXECUTABLE platoon_multi_distance_estimator
)

# LiDAR vehicle detection target
add_library(lidar_vehicle_detection_node SHARED
  src/lidar_vehicle_detection.cpp
)
target_include_directories(lidar_vehicle_detection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(lidar_vehicle_detection_node
  rclcpp
  rclcpp_components
  geometry_msgs
  loco_framework
)
target_link_libraries(lidar_vehicle_detection_node
  Eigen3::Eigen
)
rclcpp_components_register_node(lidar_vehicle_detection_node
  PLUGIN "fake_vehicle_detection::LidarVehicleDetection"
  EXECUTABLE lidar_vehicle_detection
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
# target_compile_definitions(fake_vehicle_detection PRIVATE "fake_vehicle_detection_BUILDING_LIBRARY")

# Install node libraries
install(TARGETS
  platoon_distance_estimator_node
  platoon_multi_distance_estimator_node
  lidar_vehicle_detection_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install node executables
install(TARGETS
  platoon_distance_estimator
  platoon_multi_distance_estimator
  lidar_vehicle_detection
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_libraries(
  platoon_distance_estimator_node
  platoon_multi_distance_estimator_node
  lidar_vehicle_detection_node
)

ament_package()
