cmake_minimum_required(VERSION 3.8)
project(multirotor_cmd)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(multirotor_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OMPL REQUIRED IMPORTED_TARGET ompl)
set_property(TARGET PkgConfig::OMPL PROPERTY INTERFACE_COMPILE_OPTIONS "")
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)

add_executable(position_cmd src/position_cmd.cpp)
add_executable(planning src/planning.cpp)
set_target_properties(planning PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

target_include_directories(position_cmd PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(planning PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(position_cmd
  Eigen3::Eigen
)
target_link_libraries(planning Eigen3::Eigen yaml-cpp PkgConfig::OMPL PkgConfig::IPOPT)

ament_target_dependencies(position_cmd
  rclcpp
  multirotor_interfaces
)
ament_target_dependencies(planning rclcpp multirotor_interfaces ament_index_cpp)

install(TARGETS
  position_cmd
  planning
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
