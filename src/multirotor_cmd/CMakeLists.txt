cmake_minimum_required(VERSION 3.8)
project(multirotor_cmd)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(multirotor_interfaces REQUIRED)

add_executable(position_cmd src/position_cmd.cpp)

ament_target_dependencies(position_cmd
  rclcpp
  multirotor_interfaces
)

install(TARGETS
  position_cmd
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()