#!/usr/bin/env python

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2

def publish_last_pointcloud(bag_file, topic_name, publish_topic):
    rospy.init_node('publish_last_pointcloud', anonymous=True)
    pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10)

    last_msg = None
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            last_msg = msg

    if last_msg:
        rospy.loginfo("Publishing the last PointCloud2 message from the bag")
        pub.publish(last_msg)
    else:
        rospy.loginfo("No PointCloud2 messages found in the bag on the given topic")

if __name__ == '__main__':
    bag_file = 'Raw_Point_Cloud.bag'  # Replace with your bag file path
    topic_name = '/cloud_map'   # Replace with your point cloud topic
    publish_topic = '/output_pointcloud'    # Replace with your desired publish topic

    try:
        publish_last_pointcloud(bag_file, topic_name, publish_topic)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
