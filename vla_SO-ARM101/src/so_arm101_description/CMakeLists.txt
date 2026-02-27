cmake_minimum_required(VERSION 3.10.2)
# Source: adapted from MuammerBay/SO-ARM_ROS2_URDF CMakeLists.txt
project(so_arm101_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

# Install URDF and xacro files
install(
	DIRECTORY urdf/
	DESTINATION share/${PROJECT_NAME}/urdf
	FILES_MATCHING
		PATTERN "*.urdf"
		PATTERN "*.xacro"
)

# Install mesh, config, launch, and worlds directories
install(
	DIRECTORY
		meshes
		config
		launch
		worlds
	DESTINATION share/${PROJECT_NAME}
	OPTIONAL
)

ament_package()
