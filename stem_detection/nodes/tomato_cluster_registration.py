#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Quaternion, Twist, Vector3, PoseArray
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np

from ifa_teamg.srv import perdetectService, perdetectServiceResponse

marker_pub, pose_pub, nav_pub, tf_buffer = None, None, None, None

def marker_cb(data):
    global marker_pub
    marker_pub.publish(data)

clusters, count, cluster_pub = [], [], None
cluster_thresh = 0.25
min_dets = 15
max_depth = 1.5 #1.5 for flower

published = []

register_cluster = True

def pose_cb(data):
    global pose_pub, clusters, count, cluster_thresh, published, max_depth
    pose_pub.publish(data)

    global register_cluster
    if register_cluster==False:
        return

    try:
        transform = tf_buffer.lookup_transform('odom',
                                   # source frame:
                                   data.header.frame_id,
                                   # get the tf at the time the pose was valid
                                   data.header.stamp,
                                   #rospy.Time(0),
                                   # wait for at most 1 second for transform, otherwise throw
                                   rospy.Duration(1.0))
    except Exception as e:
        print(e)
        return

    new_poses = []
    for p in data.poses:
        dist = np.sqrt(p.position.x**2 + p.position.y**2 + p.position.z**2)
        if(dist>max_depth):
            continue
        #if(p.position.z>max_depth):
        #    continue
        pose_stamped = PoseStamped()
        pose_stamped.header = data.header
        pose_stamped.pose = p
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        new_poses.append(pose_transformed)

    if len(new_poses) == 0:
        return

    print("New Poses")
    print(new_poses)

    if len(clusters)==0:
        clusters.append(new_poses[0])
        count.append(1)
        new_poses = new_poses[1:]
        published.append(0)

    for p in new_poses:
        new_cluster =True
        for i,c in enumerate(clusters):
            if calcDist(p,c) < cluster_thresh:
                new_cluster = False
                count[i] += 1
                print(calcDist(p,c))
                break
        if new_cluster==True:
            clusters.append(p)
            count.append(1)
            published.append(0)

    publishClusters()
                
def calcDist(p1, p2):
    return np.sqrt((p1.pose.position.x - p2.pose.position.x)**2 + (p1.pose.position.y - p2.pose.position.y)**2 + (p1.pose.position.z - p2.pose.position.z)**2)

cluster_pose_array = None

def publishClusters():
    global clusters, count, cluster_pub, min_dets, nav_pub, cluster_pose_array, published

    cluster_pose_array = PoseArray()
    cluster_pose_array.header.frame_id = "odom"
    cluster_pose_array.header.stamp = rospy.Time.now()

    for i,c in enumerate(clusters):
        if count[i] >= min_dets:
            cluster_pose_array.poses.append(c.pose)
            #if published[i] == 0:
            #    nav_pub.publish(c)
            #    published[i] = 1
            nav_pub.publish(c)

    cluster_pub.publish(cluster_pose_array)

    print("*******")
    for i,c in enumerate(clusters):
        print(str(count[i]) + ": {} {} {}".format(c.pose.position.x, c.pose.position.y, c.pose.position.z))


def service_handler(req):
    global register_cluster
    if req.perdetect_flag == 2:
        register_cluster = False
    else:
        register_cluster = True

    return perdetectServiceResponse(req.perdetect_flag)


if __name__=='__main__':

    rospy.init_node('jetson_hello_interface')

    tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    marker_sub = rospy.Subscriber('/cluster_loc', MarkerArray, marker_cb, queue_size=10)
    pose_sub = rospy.Subscriber('/detected_pose_array', PoseArray, pose_cb, queue_size=10)

    marker_pub = rospy.Publisher('/cluster_loc1', MarkerArray, queue_size=10, latch=True)
    pose_pub = rospy.Publisher('/detected_pose_array1', PoseArray, queue_size=10, latch=True)

    cluster_pub = rospy.Publisher('/clusters', PoseArray, queue_size=10)
    nav_pub = rospy.Publisher('/detections', PoseStamped, queue_size=10)

    s = rospy.Service('bp_to_perception_startstop', perdetectService, service_handler)


    rospy.spin()

    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        if cluster_pose_array != None:
            print("Yes")
            cluster_pub.publish(cluster_pose_array)
        r.sleep()

    
