"""Launch the HummingBird position command simulation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    path = LaunchConfiguration("path")
    plant = Node(package="plant", executable="plant", name="plant", output="screen")
    wrench_controller = Node(
        package="multirotor_controller", executable="wrench_controller",
        name="wrench_controller", output="screen",
    )
    allocator_controller = Node(
        package="multirotor_controller", executable="allocator_controller",
        name="allocator_controller", output="screen",
    )
    position_cmd = Node(
        package="multirotor_cmd", executable="position_cmd",
        name="position_cmd", output="screen", parameters=[{"path": path}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("path", default_value="att",
                              choices=["pos", "att", "step_att", "circle_demo", "demo_straight"]),
        plant,
        RegisterEventHandler(OnProcessStart(
            target_action=plant, on_start=[wrench_controller, allocator_controller],
        )),
        RegisterEventHandler(OnProcessStart(
            target_action=wrench_controller, on_start=[position_cmd],
        )),
    ])
