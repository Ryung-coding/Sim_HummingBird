from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def arg(context, name):
    return LaunchConfiguration(name).perform(context)


def farg(context, name):
    return float(arg(context, name))


def launch_setup(context, *args, **kwargs):
    plant = Node(
        package="plant",
        executable="plant",
        name="plant",
        output="screen",
        parameters=[
            {
                "theta_servo_kp": farg(context, "theta_servo_kp"),
                "theta_servo_kv": farg(context, "theta_servo_kv"),
                "phi_servo_kp": farg(context, "phi_servo_kp"),
                "phi_servo_kv": farg(context, "phi_servo_kv"),
            }
        ]
    )

    wrench_controller = Node(
        package="multirotor_controller",
        executable="wrench_controller",
        name="wrench_controller",
        output="screen",
        parameters=[
            {
                "kp_pos": [
                    farg(context, "kp_pos_x"),
                    farg(context, "kp_pos_y"),
                    farg(context, "kp_pos_z"),
                ],
                "ki_pos": [
                    farg(context, "ki_pos_x"),
                    farg(context, "ki_pos_y"),
                    farg(context, "ki_pos_z"),
                ],
                "kd_pos": [
                    farg(context, "kd_pos_x"),
                    farg(context, "kd_pos_y"),
                    farg(context, "kd_pos_z"),
                ],
                "i_min_pos": farg(context, "i_min_pos"),
                "i_max_pos": farg(context, "i_max_pos"),
                "kp_att": [
                    farg(context, "kp_att_roll"),
                    farg(context, "kp_att_pitch"),
                    farg(context, "kp_att_yaw"),
                ],
                "ki_att": [
                    farg(context, "ki_att_roll"),
                    farg(context, "ki_att_pitch"),
                    farg(context, "ki_att_yaw"),
                ],
                "kd_att": [
                    farg(context, "kd_att_roll"),
                    farg(context, "kd_att_pitch"),
                    farg(context, "kd_att_yaw"),
                ],
                "i_min_att": farg(context, "i_min_att"),
                "i_max_att": farg(context, "i_max_att"),
            }
        ]
    )

    allocator_controller = Node(
        package="multirotor_controller",
        executable="allocator_controller",
        name="allocator_controller",
        output="screen",
        parameters=[
            {
                "allocation_mode": arg(context, "allocation_mode")
            }
        ]
    )

    position_cmd = Node(
        package="multirotor_cmd",
        executable="position_cmd",
        name="position_cmd",
        output="screen"
    )

    tracking_report = Node(
        package="plant",
        executable="tracking_report",
        name="tracking_report",
        output="screen",
        parameters=[
            {
                "duration_sec": farg(context, "duration_sec")
            }
        ]
    )

    start_controllers_after_plant = RegisterEventHandler(
        OnProcessStart(
            target_action=plant,
            on_start=[
                wrench_controller,
                allocator_controller
            ]
        )
    )

    start_cmd_after_allocator = RegisterEventHandler(
        OnProcessStart(
            target_action=allocator_controller,
            on_start=[
                position_cmd,
                tracking_report
            ]
        )
    )

    shutdown_after_report = RegisterEventHandler(
        OnProcessExit(
            target_action=tracking_report,
            on_exit=[
                Shutdown(reason="30 s tracking report finished")
            ]
        )
    )

    return [
        plant,
        start_controllers_after_plant,
        start_cmd_after_allocator,
        shutdown_after_report
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument("allocation_mode", default_value="A"),
        DeclareLaunchArgument("theta_servo_kp", default_value="12.0"),
        DeclareLaunchArgument("theta_servo_kv", default_value="4.0"),
        DeclareLaunchArgument("phi_servo_kp", default_value="12.0"),
        DeclareLaunchArgument("phi_servo_kv", default_value="4.0"),
        DeclareLaunchArgument("kp_pos_x", default_value="100.0"),
        DeclareLaunchArgument("kp_pos_y", default_value="100.0"),
        DeclareLaunchArgument("kp_pos_z", default_value="100.0"),
        DeclareLaunchArgument("ki_pos_x", default_value="1.0"),
        DeclareLaunchArgument("ki_pos_y", default_value="1.0"),
        DeclareLaunchArgument("ki_pos_z", default_value="1.0"),
        DeclareLaunchArgument("kd_pos_x", default_value="40.0"),
        DeclareLaunchArgument("kd_pos_y", default_value="40.0"),
        DeclareLaunchArgument("kd_pos_z", default_value="40.0"),
        DeclareLaunchArgument("i_min_pos", default_value="-50.0"),
        DeclareLaunchArgument("i_max_pos", default_value="50.0"),
        DeclareLaunchArgument("kp_att_roll", default_value="100.0"),
        DeclareLaunchArgument("kp_att_pitch", default_value="100.0"),
        DeclareLaunchArgument("kp_att_yaw", default_value="40.0"),
        DeclareLaunchArgument("ki_att_roll", default_value="1.0"),
        DeclareLaunchArgument("ki_att_pitch", default_value="1.0"),
        DeclareLaunchArgument("ki_att_yaw", default_value="1.0"),
        DeclareLaunchArgument("kd_att_roll", default_value="20.0"),
        DeclareLaunchArgument("kd_att_pitch", default_value="20.0"),
        DeclareLaunchArgument("kd_att_yaw", default_value="20.0"),
        DeclareLaunchArgument("i_min_att", default_value="-10.0"),
        DeclareLaunchArgument("i_max_att", default_value="10.0"),
        DeclareLaunchArgument("duration_sec", default_value="30.0"),
    ]

    return LaunchDescription([
        *launch_args,
        OpaqueFunction(function=launch_setup),
    ])
