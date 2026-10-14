from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    environment_file = LaunchConfiguration("environment_file")
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="position_cmd",
        choices=["position_cmd", "planning"]
    )
    environment_arg = DeclareLaunchArgument(
        "environment_file",
        default_value=PathJoinSubstitution([FindPackageShare("plant"), "environment.yaml"])
    )

    plant = Node(
        package="plant",
        executable="plant",
        name="plant",
        output="screen",
        parameters=[{"environment_file": environment_file}],
    )

    wrench_controller = Node(
        package="multirotor_controller",
        executable="wrench_controller",
        name="wrench_controller",
        output="screen",
    )

    allocator_controller = Node(
        package="multirotor_controller",
        executable="allocator_controller",
        name="allocator_controller",
        output="screen",
    )

    position_cmd = Node(
        package="multirotor_cmd",
        executable="position_cmd",
        name="position_cmd",
        output="screen",
        condition=IfCondition(PythonExpression(["'", mode, "' == 'position_cmd'"]))
    )

    planning = Node(
        package="multirotor_cmd",
        executable="planning",
        name="planning",
        output="screen",
        parameters=[{"environment_file": environment_file}],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'planning'"]))
    )

    start_controllers_after_plant = RegisterEventHandler(
        OnProcessStart(
            target_action=plant,
            on_start=[
                wrench_controller,
                allocator_controller,
            ]
        )
    )

    start_cmd_after_wrench = RegisterEventHandler(
        OnProcessStart(
            target_action=wrench_controller,
            on_start=[
                position_cmd,
                planning,
            ]
        )
    )

    return LaunchDescription([
        mode_arg,
        environment_arg,
        plant,
        start_controllers_after_plant,
        start_cmd_after_wrench
    ])
