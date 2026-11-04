"""HummingBird simulation, with optional incremental KICK mapping and planning."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    viewer = LaunchConfiguration("viewer")
    environment_file = LaunchConfiguration("environment_file")
    trajectory = LaunchConfiguration("trajectory")
    mapping_sensors = LaunchConfiguration("mapping_sensors")
    state_topic = LaunchConfiguration("state_topic")
    wait_for_slam = LaunchConfiguration("wait_for_slam")
    mapping = IfCondition(PythonExpression(["'", mode, "' in ('map', 'slam', 'kick')"]))
    simple_cmd = IfCondition(PythonExpression(["'", mode, "' in ('position_cmd', 'map', 'slam')"]))
    kick = IfCondition(PythonExpression(["'", mode, "' == 'kick'"]))
    slam_feedback = IfCondition(PythonExpression(["'", mode, "' in ('slam', 'kick')"]))
    mapping_viewer = IfCondition(PythonExpression(["'", mode, "' in ('map', 'slam', 'kick') and '", viewer, "' == 'true'"]))

    plant = Node(
        package="plant", executable="plant", name="plant", output="screen",
        parameters=[{"environment_file": environment_file, "publish_mapping_sensors": mapping_sensors}],
    )
    wrench_controller = Node(
        package="multirotor_controller", executable="wrench_controller", name="wrench_controller",
        output="screen", parameters=[{"state_topic": state_topic}],
    )
    allocator_controller = Node(
        package="multirotor_controller", executable="allocator_controller", name="allocator_controller", output="screen",
    )
    position_cmd = Node(
        package="multirotor_cmd", executable="position_cmd", name="position_cmd", output="screen",
        parameters=[{"trajectory": trajectory, "wait_for_slam": wait_for_slam}], condition=simple_cmd,
    )
    planning = Node(
        package="multirotor_cmd", executable="planning", name="planning", output="screen",
        parameters=[{"environment_file": environment_file}],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'planning'"])),
    )
    kick_planning = Node(
        package="plant", executable="kick_planning", name="kick_planning", output="screen",
        parameters=[{"environment_file": environment_file}], condition=kick,
    )
    camera = Node(
        package="plant", executable="map_camera", name="map_camera", output="screen",
        parameters=[{"environment_file": environment_file}], condition=mapping,
    )
    rgbd_sync = Node(
        package="rtabmap_sync", executable="rgbd_sync", name="rgbd_sync", output="screen",
        parameters=[{"approx_sync": False}],
        remappings=[("rgb/image", "/camera/rgb/image_raw"),
                    ("depth/image", "/camera/depth/image_raw"),
                    ("rgb/camera_info", "/camera/rgb/camera_info"),
                    ("rgbd_image", "/camera/rgbd_image")], condition=mapping,
    )
    odom = Node(
        package="rtabmap_odom", executable="rgbd_odometry", name="rgbd_odometry", output="screen",
        parameters=[{"frame_id": "base_link", "odom_frame_id": "odom", "publish_tf": True,
                     "subscribe_rgbd": True, "wait_imu_to_init": True,
                     "always_check_imu_tf": False, "approx_sync": False,
                     "Vis/MinInliers": "10", "Vis/FeatureType": "8"}],
        remappings=[("rgbd_image", "/camera/rgbd_image"), ("imu", "/imu/data"),
                    ("odom", "/odom")], condition=mapping,
    )
    slam = Node(
        package="rtabmap_slam", executable="rtabmap", name="rtabmap", namespace="rtabmap", output="screen",
        parameters=[{"frame_id": "base_link", "map_frame_id": "map", "odom_frame_id": "odom",
                     "subscribe_rgb": False, "subscribe_depth": False, "subscribe_rgbd": True,
                     "subscribe_odom_info": True, "approx_sync": False,
                     "database_path": "/tmp/hb_mapchk_rtabmap.db", "map_always_update": True,
                     "Grid/3D": "true", "Grid/RayTracing": "true",
                     "Grid/CellSize": "0.20", "Grid/RangeMax": "8.0",
                     "Grid/GroundIsObstacle": "true", "RGBD/LinearUpdate": "0.15",
                     "RGBD/AngularUpdate": "0.05", "Rtabmap/DetectionRate": "4"}],
        remappings=[("rgbd_image", "/camera/rgbd_image"), ("odom", "/odom"),
                    ("odom_info", "/odom_info"), ("imu", "/imu/data")],
        arguments=["--delete_db_on_start"], condition=mapping,
    )
    edt = Node(package="plant", executable="map_edt", name="map_edt", output="screen", condition=mapping)
    slam_state = Node(
        package="plant", executable="slam_state", name="slam_state", output="screen",
        parameters=[{"environment_file": environment_file}], condition=slam_feedback,
    )
    viewers = [Node(
        package="rviz2", executable="rviz2", name=f"{name}_viewer", output="screen",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("plant"), "rviz", f"{name}_viewer.rviz"])],
        condition=mapping_viewer,
    ) for name in ("rtab", "octomap", "edt")]

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="position_cmd",
                              choices=["position_cmd", "planning", "map", "slam", "kick"]),
        DeclareLaunchArgument("viewer", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("trajectory", default_value=PythonExpression(["'mapCHK' if '", mode, "' in ('map', 'slam') else 'attPath'"])),
        DeclareLaunchArgument("mapping_sensors", default_value=PythonExpression(["'", mode, "' in ('map', 'slam', 'kick')"])),
        DeclareLaunchArgument("state_topic", default_value=PythonExpression(["'/slam_state' if '", mode, "' in ('slam', 'kick') else '/multirotor_state'"])),
        DeclareLaunchArgument("wait_for_slam", default_value=PythonExpression(["'", mode, "' == 'slam'"])),
        DeclareLaunchArgument("environment_file", default_value=PythonExpression([
            "'", FindPackageShare("plant"), "/environment_map.yaml' if '", mode,
            "' in ('map', 'slam', 'kick') else '", FindPackageShare("plant"), "/environment.yaml'"])),
        plant,
        RegisterEventHandler(OnProcessStart(target_action=plant, on_start=[wrench_controller, allocator_controller])),
        RegisterEventHandler(OnProcessStart(target_action=wrench_controller,
                                            on_start=[position_cmd, planning, kick_planning])),
        camera, rgbd_sync, odom, slam, edt, slam_state, *viewers,
    ])
