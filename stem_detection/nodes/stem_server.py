#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import ros_numpy
from ros_numpy.point_cloud2 import split_rgb_field, get_xyz_points

import cv2

from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros
import tf2_geometry_msgs

from ifa_teamg.srv import perService, perServiceResponse

stem_pos = []
clusters = []
processed_cloud = False
stem_loc = PoseStamped()

full_cloud, curr_cloud = [], []
global_clusters, global_stems = [], []

no_stem_pts = False

def callback(data):

    print("*****")

    pc2_numpy = ros_numpy.numpify(data)
    rgb = pc2_numpy['rgb']
    pc2_numpy = split_rgb_field(pc2_numpy)
    points = get_xyz_points(pc2_numpy)
    # print(points.shape)

    # Tf to odom
    try:
        transform = tf_buffer.lookup_transform("odom",
                                    # source frame:
                                    data.header.frame_id,
                                    # get the tf at the time the pose was valid
                                    rospy.Time(0),
                                    # wait for at most 1 second for transform, otherwise throw
                                    rospy.Duration(1.0))
    except Exception as e:
        print(e)
        return

    H_mat = ros_numpy.numpify(transform.transform)
    #print("H mat")
    #print(H_mat)

    homo_pts = np.hstack((points, np.ones((points.shape[0],1))))
    odom_pts = homo_pts.dot(H_mat.T)
    odom_pts = odom_pts/odom_pts[:,-1].reshape(-1,1)
    #print(odom_pts.shape)

    # Reverse sorted pts in odom 
    sorted_idx = (-odom_pts[:,2]).argsort()
    sorted_pts = odom_pts[sorted_idx]
    # print(sorted_pts[0], sorted_pts[-1])
    # stemX, stemY, stemZ, _ =  sorted_pts.mean(axis=0)

    # Stem location based on removing red pts using HSV filter
    r_sorted = pc2_numpy['r'][sorted_idx]
    g_sorted = pc2_numpy['g'][sorted_idx]
    b_sorted = pc2_numpy['b'][sorted_idx]
    dummy_img = (np.stack((r_sorted, g_sorted, b_sorted)).T).reshape(-1,1,3)

    global no_stem_pts 
    if dummy_img.shape[0] == 0:
        no_stem_pts = True
        return
    
    red_mask = detect_color(dummy_img)
    red_pixels = (red_mask!=0)[:,0]
    idx = np.where(red_pixels==True)[0]

    if idx.shape[0] == 0:
        no_stem_pts = True
        return

    stem_pc2_pts = sorted_pts[:idx.min()]
    print(odom_pts.shape,stem_pc2_pts.shape)

    if stem_pc2_pts.shape[0] != 0:
        stemX, stemY, stemZ, _ =  stem_pc2_pts.mean(axis=0)
    else:
        no_stem_pts = True
        return

    # if stem_pc2_pts.shape[0] < 100:
    #     stemZ += 0.2

    p = Pose()
    p.position.x = stemX
    p.position.y = stemY
    p.position.z = stemZ
    p.orientation.x = 0.0
    p.orientation.y = 1.0
    p.orientation.z = 0.0
    p.orientation.w = 0.0

    global stem_loc
    stem_loc.header.stamp = rospy.Time.now()
    stem_loc.header.frame_id = "odom"
    stem_loc.pose = p

    odom_pts[:,-1] = rgb
    global curr_cloud
    curr_cloud = odom_pts

    global processed_cloud
    processed_cloud = True
	

def detect_color(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    '''Red'''
    # Range for lower red
    red_lower = np.array([0,100,20])
    red_upper = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([160,100,20])
    red_upper = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2
    return mask_red


def cluster_cb(data):
    global clusters
    clusters = []
    for marker in data.markers:
        clusters.append(marker.pose)


def tf_to_target(poseStamped, target_frame, source_frame):
    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
    except Exception as e:
        print(e)
        return
    pose_transformed = tf2_geometry_msgs.do_transform_pose(poseStamped, transform)
    return pose_transformed


def service_handler(req):
    global clusters, processed_cloud, stem_loc, global_clusters, global_stems, no_stem_pts

    print("Received request")
    processed_cloud = False
    
    pose_transformed = tf_to_target(req.pose, 'head_camera_color_optical_frame', req.pose.header.frame_id)
    reqX = pose_transformed.pose.position.x
    reqY = pose_transformed.pose.position.y
    reqZ = pose_transformed.pose.position.z

    min_dist = np.inf
    matched_pose = None
    print("Detected clusters")
    print(len(clusters))
    for c in clusters:
        dist = np.sqrt( (reqX - c.position.x)**2 + (reqY - c.position.y)**2 + (reqZ - c.position.z)**2)
        print("Distance to cluster")
        print(dist)
        if dist<min_dist and dist<0.3:
            matched_pose = c
            break
    
    if matched_pose==None:
        print("No matching cluster found, false positive")
        return perServiceResponse(0, stem_loc.pose)

    p = PoseStamped()
    p.pose = matched_pose
    p.header.frame_id = 'head_camera_color_optical_frame'
    p.header.stamp = rospy.Time.now()

    global_clusters.append(tf_to_target(p, 'odom', p.header.frame_id))
    publish_stems_and_clusters()

    stem_pos = []
    r = rospy.Rate(100)

    while len(stem_pos) < 5:
        while not processed_cloud:
            pub1.publish(p)
            r.sleep()

        if no_stem_pts:
            print("No stem points in point cloud")
            no_stem_pts = False
            return perServiceResponse(0, stem_loc.pose)

        pose_i = np.array([stem_loc.pose.position.x,stem_loc.pose.position.y,stem_loc.pose.position.z])
        if not np.isfinite(pose_i).all():
            continue

        stem_pos.append(pose_i)
        processed_cloud = False
        print("Added pose")

    print(np.array(stem_pos))
    stemX, stemY, stemZ = np.array(stem_pos).mean(axis=0)
    stem_loc.pose.position.x = stemX
    stem_loc.pose.position.y = stemY
    stem_loc.pose.position.z = stemZ

    pub.publish(stem_loc)

    global_stems.append(stem_loc)
    # global_clusters.append(tf_to_target(p, 'odom', p.header.frame_id))
    publish_stems_and_clusters()
    publish_cloud()
    
    return perServiceResponse(1, stem_loc.pose)


def publish_stems_and_clusters():

    global global_clusters, global_stems

    cluster_pose_array = PoseArray()
    cluster_pose_array.header.frame_id = "odom"
    cluster_pose_array.header.stamp = rospy.Time.now()

    for c in global_clusters:
        cluster_pose_array.poses.append(c.pose)


    stem_marker_array = MarkerArray()
    for idx, stem in enumerate(global_stems):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.id = idx
        marker.type = 2
        marker.action = marker.ADD
        marker.pose = stem.pose
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = 0,0,0,1
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.frame_locked = False

        stem_marker_array.markers.append(marker)   

    pub2.publish(stem_marker_array)
    pub3.publish(cluster_pose_array)


def publish_cloud():
    global curr_cloud, full_cloud

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 16, PointField.FLOAT32, 1),]

    
    if len(full_cloud) == 0:
        full_cloud = curr_cloud
    else:
        print(full_cloud.shape, curr_cloud.shape)
        full_cloud = np.vstack((full_cloud, curr_cloud))

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "odom"
    odom_pc2 = pc2.create_cloud(header, fields, full_cloud)
    pub4.publish(odom_pc2)



if __name__=='__main__':
    rospy.init_node('pc2_to_numpy', anonymous=False)

    rospy.Subscriber("cluster_loc1", MarkerArray, cluster_cb, queue_size=1)
    rospy.Subscriber("extracted_cloud", PointCloud2, callback, queue_size=1)

    pub = rospy.Publisher('/stem_loc', PoseStamped, queue_size=1)
    pub1 = rospy.Publisher('exctraction_pt', PoseStamped, queue_size=1)
    pub2 = rospy.Publisher('/global_stems', MarkerArray, queue_size=10)
    pub3 = rospy.Publisher('/global_clusters', PoseArray, queue_size=10)
    pub4 = rospy.Publisher('global_cluster_cloud', PointCloud2, queue_size=1)
   
    tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    s = rospy.Service('bp_to_perception', perService, service_handler)

    rospy.spin()
