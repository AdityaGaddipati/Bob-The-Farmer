#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2, Image, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import ros_numpy
from ros_numpy.point_cloud2 import split_rgb_field, get_xyz_points
from ros_numpy.image import image_to_numpy, numpy_to_image

import tf2_ros
import tf2_geometry_msgs

# import matplotlib.pyplot as plt
import cv2

pt_2d = None
stem_pts = []

def create_cloud(pts, rgb):
	fields = [PointField('x', 0, PointField.FLOAT32, 1),
			  PointField('y', 4, PointField.FLOAT32, 1),
			  PointField('z', 8, PointField.FLOAT32, 1),
			  PointField('rgb', 16, PointField.FLOAT32, 1),
			]

	print(type(rgb[0]))
	pts[:,-1] = rgb
	header = Header()
	header.stamp = rospy.Time.now()
	header.frame_id = "odom"
	odom_pc2 = pc2.create_cloud(header, fields, pts)
	pc2_pub.publish(odom_pc2)

def callback(data):
	print("*****")

	pc2_numpy = ros_numpy.numpify(data)

	rgb = pc2_numpy['rgb']

	pc2_numpy = split_rgb_field(pc2_numpy)
	

	points = get_xyz_points(pc2_numpy)
	print(points.shape)

	# # Sorted pts in camera frame's X direction
	# sorted_idx = points[:,0].argsort()
	# sorted_pts = points[sorted_idx]

	# # Taking avg of 5 pts at index
	# global pt_2d, stem_pts 
	# index = 0
	# if len(stem_pts) < 5:
	# 	stem_pts.append(sorted_pts[index])
	# 	return
	# else:
	# 	stem_pts.pop(0)
	# 	stem_pts.append(sorted_pts[index])
	# 	stemX, stemY, stemZ = np.mean(np.array(stem_pts), axis=0)
	# 	print(stemX, stemY, stemZ)


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
	# stemX, stemY, stemZ, _ =  sorted_pts[:-2000].mean(axis=0)


	# Red filter
	r_sorted = pc2_numpy['r'][sorted_idx]
	g_sorted = pc2_numpy['g'][sorted_idx]
	b_sorted = pc2_numpy['b'][sorted_idx]
	dummy_img = (np.stack((r_sorted, g_sorted, b_sorted)).T).reshape(-1,1,3)
	red_mask = detect_color(dummy_img)
	red_pixels = (red_mask!=0)[:,0]
	idx = np.where(red_pixels==True)[0]
	stem_pc2_pts = sorted_pts[:idx.min()]
	print(odom_pts.shape,stem_pc2_pts.shape)
	stemX, stemY, stemZ, _ =  stem_pc2_pts.mean(axis=0)

	if stem_pc2_pts.shape[0] < 100:
		stemZ += 0.1


	p = Pose()
	p.position.x = stemX
	p.position.y = stemY
	p.position.z = stemZ
	p.orientation.x = 0.0
	p.orientation.y = 1.0
	p.orientation.z = 0.0
	p.orientation.w = 0.0

	loc = PoseStamped()
	loc.header.stamp = rospy.Time.now()
	# loc.header.frame_id = data.header.frame_id
	loc.header.frame_id = "odom"
	loc.pose = p

	pub.publish(loc)

	#print(proj_mat)
	pt_3d = np.array([stemX,stemY,stemZ,1])
	pt_2d = proj_mat.dot(pt_3d)
	pt_2d = (pt_2d/pt_2d[2]).astype(np.int)
	#print(pt_2d/pt_2d[2])

	create_cloud(odom_pts, rgb)

	# # Variance plot
	# N = sorted_pts.shape[0]
	# iters = []
	# std = []
	# for i in np.arange(100,N,100):
	# 	stdX = np.var(sorted_pts[i-100:i,0])
	# 	stdY = np.var(sorted_pts[i-100:i,1])
	# 	iters.append(i)
	# 	std.append(stdX+stdY)

	# fig, ax = plt.subplots()
	# ax.plot(iters,std)
	# plt.show()

	# #H_mat tf debug
	# loc.pose.position.x = points[0,0]
	# loc.pose.position.y = points[0,1]
	# loc.pose.position.z = points[0,2]
	# loc.pose.orientation.x = 0.0
	# loc.pose.orientation.y = 0.0
	# loc.pose.orientation.z = 0.0
	# loc.pose.orientation.w = 1.0
	# pose_transformed = tf2_geometry_msgs.do_transform_pose(loc, transform)
	# print(loc)
	# print(pose_transformed)
	# print(odom_pts[0]/odom_pts[0,3])
	
	# # Red filter
	# dummy_img = (np.stack((pc2_numpy['r'], pc2_numpy['g'], pc2_numpy['b'])).T).reshape(-1,1,3)
	# red_mask = detect_color(dummy_img)
	# print(red_mask.shape)
	# idx = (red_mask!=0)[:,0]
	# stem_pc2_pts = odom_pts[idx[:,0]]
	# print(odom_pts.shape,stem_pc2_pts.shape)

	# # stem pc2
	# pc2_numpy['x'][idx] = 0 
	# pc2_numpy['y'][idx] = 0 
	# pc2_numpy['z'][idx] = 0 

	# pc2_numpy['r'][idx] = 0 
	# pc2_numpy['g'][idx] = 0 
	# pc2_numpy['b'][idx] = 0

	# stem_pc2 = array_to_pointcloud2(pc2_numpy, rospy.Time.now(), "head_camera_color_optical_frame")
	# pc2_pub.publish(stem_pc2)


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

    #red_output = cv2.bitwise_and(image, image, mask=mask_red)
    #red_output = cv2.cvtColor(red_output, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Red", image)
    #cv2.waitKey(0)

    return mask_red


def im_cb(image):
	global pt_2d
	print("*********")
	print(image.height, image.width)
	print(pt_2d)
	np_img = image_to_numpy(image)

	#viz_image = cv2.circle(np_img, (pt_2d[0],pt_2d[1]), 20, (0,255,0), -1)
	viz_image = cv2.circle(np_img, (360-pt_2d[1]//2,pt_2d[0]//2), 7, (0,0,255), -1)
	ros_img = numpy_to_image(viz_image, "rgb8")

	image_pub.publish(ros_img)

	#fig,ax = plt.subplots(1)
	#ax.imshow(np_img)
	#plt.show()
	

if __name__=='__main__':
	rospy.init_node('pc2_to_numpy', anonymous=False)
	rospy.Subscriber("extracted_cloud", PointCloud2, callback)
	
	pub = rospy.Publisher('/stem_loc_debug', PoseStamped, queue_size=1)

	#image_sub = rospy.Subscriber("inference", Image, im_cb, queue_size=30)
	#image_sub = rospy.Subscriber("head_camera/color/image_raw", Image, im_cb, queue_size=30)
	image_pub = rospy.Publisher('viz_img', Image, queue_size=1)

	proj_mat = np.array([913.5904541015625, 0.0, 651.5320434570312, 0.0, 0.0, 913.64990234375, 371.7758483886719, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,-1)

	tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
	tf_listener = tf2_ros.TransformListener(tf_buffer)

	pc2_pub = rospy.Publisher('stem_pc2', PointCloud2, queue_size=1)

	rospy.spin()