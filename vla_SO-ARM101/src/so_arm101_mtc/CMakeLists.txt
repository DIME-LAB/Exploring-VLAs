cmake_minimum_required(VERSION 3.16)
project(so_arm101_mtc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  rclcpp
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  geometry_msgs
  shape_msgs
  tf2_msgs
  tf2_eigen
  tf2_geometry_msgs
  std_srvs
  Eigen3
)

foreach(dep IN LISTS THIS_PACKAGE_INCLUDE_DEPENDS)
  find_package(${dep} REQUIRED)
endforeach()

add_executable(mtc_pick_place_node src/mtc_pick_place_node.cpp)
ament_target_dependencies(mtc_pick_place_node ${THIS_PACKAGE_INCLUDE_DEPENDS})

install(TARGETS mtc_pick_place_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
