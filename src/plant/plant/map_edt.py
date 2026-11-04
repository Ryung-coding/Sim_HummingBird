#!/usr/bin/env python3
"""Euclidean distance field over observed OctoMap voxels."""

import numpy as np
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class MapEDT(Node):
    def __init__(self):
        super().__init__("map_edt")
        self.resolution = 0.2
        self.origin = np.array([-1.4, -2.5, 0.0], dtype=np.float32)
        self.shape = (75, 26, 17)  # x, y, z; z-up world, 15.0 x 5.2 x 3.4 m
        self.empty = None
        self.occupied = None
        self.map_header = None
        self.dirty = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PointCloud2, "/rtabmap/octomap_occupied_space", self.on_occupied, qos)
        self.create_subscription(PointCloud2, "/rtabmap/octomap_empty_space", self.on_empty, qos)
        self.publisher = self.create_publisher(PointCloud2, "/map_edt/observed_distance", qos)
        self.create_timer(1.0, self.publish_edt)

    @staticmethod
    def points(msg):
        arr = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        return np.asarray(arr, dtype=np.float32).reshape(-1, 3)

    def on_occupied(self, msg):
        self.occupied = self.points(msg)
        self.map_header = msg.header
        self.dirty = True

    def on_empty(self, msg):
        self.empty = self.points(msg)
        self.dirty = True

    def voxel_indices(self, points):
        indices = np.floor((points - self.origin) / self.resolution).astype(np.int32)
        valid = np.all((indices >= 0) & (indices < self.shape), axis=1)
        return indices[valid]

    def publish_edt(self):
        if not self.dirty or self.occupied is None or self.empty is None:
            return
        self.dirty = False
        occupied = self.voxel_indices(self.occupied)
        empty = self.voxel_indices(self.empty)
        if len(occupied) == 0 or len(empty) == 0:
            return
        grid = np.zeros(self.shape, dtype=bool)
        grid[occupied[:, 0], occupied[:, 1], occupied[:, 2]] = True
        distance = distance_transform_edt(~grid, sampling=self.resolution)
        observed = np.zeros(self.shape, dtype=bool)
        observed[empty[:, 0], empty[:, 1], empty[:, 2]] = True
        observed[grid] = False
        indices = np.argwhere(observed)
        if len(indices) == 0:
            return
        xyz = self.origin + (indices.astype(np.float32) + 0.5) * self.resolution
        values = distance[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float32)
        samples = np.column_stack((xyz, values)).astype(np.float32)
        fields = [PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, name in enumerate(("x", "y", "z", "intensity"))]
        self.publisher.publish(point_cloud2.create_cloud(self.map_header, fields, samples))
        self.get_logger().info(f"OctoMap: occupied={len(occupied)}, observed_free={len(indices)}, "
                               f"EDT range={values.min():.2f}..{values.max():.2f} m", throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = MapEDT()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
