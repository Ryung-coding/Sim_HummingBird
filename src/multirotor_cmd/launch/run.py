from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    vehicle = LaunchConfiguration("vehicle")
    mode = LaunchConfiguration("mode")
    vehicle_arg = DeclareLaunchArgument(
        "vehicle",
        default_value="hummingbird",
        choices=["hummingbird", "hexa"]
    )
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="position_cmd",
        choices=["position_cmd", "planning"]
    )

    plant = Node(
        package="plant",
        executable="plant",
        name="plant",
        output="screen",
        parameters=[{"vehicle": vehicle, "mode": mode}]
    )

    wrench_controller = Node(
        package="multirotor_controller",
        executable="wrench_controller",
        name="wrench_controller",
        output="screen",
        parameters=[{"vehicle": vehicle}]
    )

    allocator_controller = Node(
        package="multirotor_controller",
        executable="allocator_controller",
        name="allocator_controller",
        output="screen",
        parameters=[{"vehicle": vehicle}]
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
        vehicle_arg,
        mode_arg,
        plant,
        start_controllers_after_plant,
        start_cmd_after_wrench
    ])
