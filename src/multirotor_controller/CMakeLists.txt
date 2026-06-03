cmake_minimum_required(VERSION 3.8)
project(multirotor_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(multirotor_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(wrench_controller src/wrench_controller.cpp)
add_executable(allocator_controller src/allocator_controller.cpp)

target_include_directories(wrench_controller PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_include_directories(allocator_controller PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(wrench_controller
  Eigen3::Eigen
)

target_link_libraries(allocator_controller
  Eigen3::Eigen
)

ament_target_dependencies(wrench_controller
  rclcpp
  multirotor_interfaces
)

ament_target_dependencies(allocator_controller
  rclcpp
  multirotor_interfaces
)

install(TARGETS
  wrench_controller
  allocator_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()