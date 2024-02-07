#!/usr/bin/env python3

import rospy
import struct
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2,PointField
import rosbag
import open3d as o3d
import numpy as np
import std_msgs.msg


def ros_to_open3d(ros_point_cloud):
    # print(ros_point_cloud.point_step,ros_point_cloud.row_step )
    points = np.array(list(pc2.read_points(ros_point_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True)))
    # Separate XYZ and RGB
    xyz = points[:, :3]
    rgb = points[:, 3]
    # Function to convert float32 RGB to three uint8 values
    def float_to_rgb(float_rgb):
        int_rgb = struct.unpack('I', struct.pack('f', float_rgb))[0]
        return ((int_rgb >> 16) & 0xFF, (int_rgb >> 8) & 0xFF, int_rgb & 0xFF)

    # Convert RGB float32 values to three uint8 values
    colors = np.array([float_to_rgb(color) for color in rgb], dtype=np.uint32) / 255.0
    # Create Open3D point cloud
    open3d_pc = o3d.geometry.PointCloud()
    open3d_pc.points = o3d.utility.Vector3dVector(xyz)
    open3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return open3d_pc


def open3d_to_ros(open3d_point_cloud, frame_id="map"):
    # Create header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    # Check if the point cloud has color information
    has_colors = open3d_point_cloud.has_colors()
    # Extract points
    points = np.asarray(open3d_point_cloud.points)
    # Create fields for PointCloud2
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
    # Add color information if available
    if has_colors:
        #retrieve color information
        fields.append(PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1))        
        colors = np.round(np.asarray(open3d_point_cloud.colors) * 255).astype(np.uint32)
        # Pack RGB values into the float
        packed_ints = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        rgb = packed_ints.view(np.float32)
        points = np.column_stack((points, rgb))
        
    print(len(points))    
    # Create PointCloud2 message
    cloud_msg = pc2.create_cloud(header, fields, points)
    return cloud_msg


def get_pointcloud(bag_file, topic_name):
    last_msg = None
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            last_msg = msg
    return last_msg


class FilterNode:
    def __init__(self):
        rospy.init_node('Filter', anonymous=True)

        self.filtered_cloud_pub = rospy.Publisher('/filtered_pointcloud', PointCloud2, queue_size=1)
        self.og_cloud_pub = rospy.Publisher('/original_pointcloud', PointCloud2, queue_size=1)
        rospy.Subscriber('/cloud_map', PointCloud2, self.pc_callback)
    
    # Store latest point cloud message
    def pc_callback(self, cloud_mesage):
        self.cloud = cloud_mesage
    
    ### MAIN FILTERING FUNCTION ###
    def filter(self):
        self.og_cloud_msg = get_pointcloud('/home/rayvo50/cv3dip/projects/project3/catkin_ws/src/pc_filter/scripts/Raw_Point_Cloud.bag', '/cloud_map')
        print(self.og_cloud_msg.width, self.og_cloud_msg.height)
        cloud = ros_to_open3d(self.og_cloud_msg)
        
        cloud = pass_through_filter(cloud, axis='z', axis_min=-0.9, axis_max=0.9)
        cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        #cloud = segment_planes(cloud) # not succesfully implemented

        # Convert back to ROS PointCloud2 format and publish
        self.filtered_cloud_msg = open3d_to_ros(cloud)
        print(self.filtered_cloud_msg.width, self.filtered_cloud_msg.height)
        rospy.loginfo("Done")

    def idle(self):
        self.filtered_cloud_pub.publish(self.filtered_cloud_msg)
        self.og_cloud_pub.publish(self.og_cloud_msg)

def pass_through_filter(pc, axis='z', axis_min=0.0, axis_max=1.0):
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors) if pc.has_colors() else None
    
    # Filter points based on the specified axis limits
    if axis == 'x':
        axis_ind = 0
    elif axis == 'y':
        axis_ind = 1
    elif axis == 'z':
        axis_ind = 2
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
        
    filtered_indices = np.where((points[:, axis_ind] >= axis_min) & (points[:, axis_ind] <= axis_max))[0]
    filtered_points = points[filtered_indices, :]
    
    # Create a new Open3D point cloud for the filtered points
    filtered_pc = o3d.geometry.PointCloud()
    filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
    if colors is not None:
        filtered_colors = colors[filtered_indices, :]
        filtered_pc.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    return filtered_pc

# FAILED IMPLEMENTATION:
def segment_planes(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, angle_threshold=np.pi/4):
    all_walls_pcd = o3d.geometry.PointCloud()

    for i in range(10):
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        if len(inliers) < 150:  # Adjust as necessary
            break

        # Calculate the angle between the plane normal and the Z-axis
        # Plane model returns (a, b, c, d) where (a, b, c) is the normal vector
        normal_vector = np.array(plane_model[:3])
        z_axis = np.array([0, 0, 1])
        angle = np.arccos(np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis)))

        # Check if the plane is vertical by comparing the angle with the threshold
        if abs(angle) > angle_threshold and abs(angle) < (np.pi - angle_threshold):
            inlier_cloud = pcd.select_by_index(inliers)
            all_walls_pcd += inlier_cloud

        # Remove the inliers from the point cloud to find the next plane
        pcd = pcd.select_by_index(inliers, invert=True)

    all_walls_pcd.paint_uniform_color([1.0, 0, 0])  # Paint all wall points red for visualization

    return all_walls_pcd


def main():

    filter = FilterNode()
    filter.filter()

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        filter.idle()
        r.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
