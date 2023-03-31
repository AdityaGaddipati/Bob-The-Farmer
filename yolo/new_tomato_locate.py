#!/usr/bin/env python3

import sys
import math
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
import torchvision.transforms as transforms

import message_filters

from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Quaternion, Twist, Vector3, PoseArray

#Visualiztion stuff
from visualization_msgs.msg import MarkerArray, Marker


class image_converter:

  def __init__(self, model, transform, device):
    self.image_pub = rospy.Publisher("/inference",Image, queue_size=10)
    self.depth_pub = rospy.Publisher("/depth_inference",Image, queue_size=10)

    #self.image_sub = message_filters.Subscriber("/camera/color/image_raw",Image)
    #self.depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
    #self.cam_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.depth_info_callback)

    self.image_sub = message_filters.Subscriber("/head_camera/color/image_raw",Image)
    self.depth_image_sub = message_filters.Subscriber("/head_camera/aligned_depth_to_color/image_raw",Image)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_image_sub],1,5,allow_headerless=True)
    self.ts.registerCallback(self.callback)

    self.cam_info_sub = rospy.Subscriber("/head_camera/aligned_depth_to_color/camera_info", CameraInfo, self.depth_info_callback)
    #self.cam_info_sub = rospy.Subscriber("/head_camera/depth/camera_info", CameraInfo, self.depth_info_callback)

    self.cluster_vis = rospy.Publisher('/cluster_loc', MarkerArray, queue_size=10)
    self.pose_pub = rospy.Publisher('/detected_pose_array', PoseArray, queue_size=10)

    self.model = model
    self.transforms = transform
    self.device = device

    # Detection confidence threshold
    self.conf_thresh = 0.7
    # Merge threshold
    self.thr = 40
    # Vis image scale
    self.scale = 50
    self.intrinsics = None
    self.cv_image = None
    self.fx = None
    self.cx = None
    self.fy = None
    self.cy = None



  def detect_color(self, image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    '''Red'''
    # Range for lower red
    red_lower = np.array([0,100,20])
    red_upper = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)

    # Range for upper red
    red_lower = np.array([160,100,20])
    red_upper = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2

    #red_output = cv2.bitwise_and(image, image, mask=mask_red)
    #red_output = cv2.cvtColor(red_output, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Red", image)
    #cv2.waitKey(0)

    return mask_red


  def depth_info_callback(self, cameraInfo):
    self.fx = cameraInfo.K[0]
    self.fy = cameraInfo.K[4]
    self.cx = cameraInfo.K[2]
    self.cy = cameraInfo.K[5]



  def draw_boxes(self, boxes, color):
    #colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,0,255)]
    for i,box in enumerate(boxes):
      cv2.rectangle(self.cv_image, (box[0], box[1]), (box[2], box[3]), color, 2)


  def bbox_to_XYZ(self, bbox):

    self.red = self.detect_color(self.color_image)
    
    XYZ = []
    valid_bb = []
    for bb in bbox:
      cx = int((bb[0]+bb[2])/2)
      cy = int((bb[1]+bb[3])/2)

      #Apply red filter check
      if self.red[cy,cx]==0:
        continue

      Z = self.depth_data[cy, cx][0]
      if(Z==0):
        continue

      X = ((cx - self.cx) / self.fx) * Z
      Y = ((cy - self.cy) / self.fy) * Z

      XYZ.append([X,Y,Z])
      valid_bb.append(bb)

    XYZ = np.array(XYZ)/1000
    valid_bb = np.array(valid_bb).astype(np.int)
    print("XYZ")
    print(XYZ)
    return XYZ, np.array(valid_bb)

  def cluster_XYZ(self, xyz, valid_bb):
    if len(xyz)==0:
      return []

    distance = np.sqrt((xyz**2).sum(axis=1))
    # print("Distance")
    # print(distance)

    xyz = xyz[distance<1.5]
    valid_bb = valid_bb[distance<1.5]

    if len(xyz)==0:
      return []

    clusters, cluster_bbox = self.cluster_(xyz, valid_bb)

    print("Clusters")
    print(len(clusters))
    print(len(cluster_bbox))

    cluster_box = []
    color = [(0,0,255), (255,0,0), (255,0,255)]
    for i,c in enumerate(clusters):
      boxes = np.array(cluster_bbox[i])
      #self.draw_boxes(boxes, color[0])

      xmin, ymin = boxes[:,:2].min(axis=0)
      xmax, ymax = boxes[:,2:].max(axis=0)
      
      cluster_box.append([xmin, ymin, xmax, ymax])

    self.draw_boxes(np.array(cluster_box), (0,255,0))

    return clusters


  def cluster_(self, points, boxes):
    cluster_thresh = 0.1

    clusters = []
    cluster_bbox = []

    clusters.append([points[0]])
    cluster_bbox.append([boxes[0]])

    points = points[1:]
    boxes = boxes[1:]

    for i, p in enumerate(points):

      added = False
      for j, clust_arr in enumerate(clusters):
       
        for c in clust_arr:
          dist = np.sqrt(((p-c)**2).sum())
          if dist < cluster_thresh:
            clusters[j].append(p)
            cluster_bbox[j].append(boxes[i])
            added = True
            break

        if added:
          break

      if not added:
        clusters.append([p])
        cluster_bbox.append([boxes[i]])

    return clusters, cluster_bbox


  def callback(self, rgb_image, depth_image):
    # print(rgb_image.header.frame_id, depth_image.header.frame_id)
    # print(rgb_image.height, rgb_image.width)

    # try:
    self.cv_image = self.imgmsg_to_cv2(rgb_image)
    self.color_image = self.cv_image.copy()
    self.depth_image = depth_image

    # print("size of cv image", self.cv_image.shape)

    bbox = self.run_inference(np.asarray(self.cv_image))
    
    if(bbox is None):
      print("No detections")
      return

    self.depth_data = np.frombuffer(self.depth_image.data, dtype=np.uint16).reshape(self.depth_image.height, self.depth_image.width, -1)
    
    xyz, valid_bb = self.bbox_to_XYZ(bbox)
    clusters = self.cluster_XYZ(xyz, valid_bb)


    marker_array = MarkerArray()
    pose_array = PoseArray()
    pose_array.header = rgb_image.header

    for idx, cluster in enumerate(clusters):
      coordinates = np.array(cluster).mean(axis=0)
      print(idx, " Coordinates:", coordinates[0], coordinates[1], coordinates[2])
      if math.isnan(coordinates[2]):
          continue
      marker = Marker()
      marker.header.frame_id = rgb_image.header.frame_id
      marker.id = idx
      marker.type = 2
      marker.action = marker.ADD
      marker.pose.position.x = coordinates[0]
      marker.pose.position.y = coordinates[1]
      marker.pose.position.z = coordinates[2]
      marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = 0,0,0,1
      marker.color.r = 0
      marker.color.g = 1
      marker.color.b = 0
      marker.color.a = 1
      marker.scale.x = 0.1
      marker.scale.y = 0.1
      marker.scale.z = 0.1
      marker.frame_locked = False

      marker_array.markers.append(marker)

      pose = Pose()
      pose.position.x = coordinates[0]
      pose.position.y = coordinates[1]
      pose.position.z = coordinates[2]
      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 0,0,0,1
      pose_array.poses.append(pose)
      
    # print(len(marker_array.markers))
    if len(marker_array.markers) != 0:
        self.cluster_vis.publish(marker_array)
        self.pose_pub.publish(pose_array)

    # depth_data = cv2.circle(depth_data, c, 0, (0,255,0), 2)
    # self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_data))

    self.image_pub.publish(self.cv2_to_imgmsg(self.cv_image))
      

    # except Exception as e:
    #   print(e)


  def run_inference(self, img):
    #img = self.transforms(img)
    # print(img.shape)
    
    with torch.no_grad():
      results = self.model(img)
      results_np = results.xyxy[0].detach().cpu().numpy()

    # print("*****")
    #print(results.xyxy[0])
    #print("\n")
    #print(results_np)
    #print("\n")

    bbox = []
    scores = []
    for r in results_np:
      bbox.append(r[0:4])
      scores.append(r[4])    

    bbox = np.array(bbox)
    scores = np.array(scores)

    #print(bbox)
    #print("\n")
    #print(scores)
    #print("\n")

    return bbox[scores>self.conf_thresh]

  def imgmsg_to_cv2(self, img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    img_opencv_rgb = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)

    #img_opencv_bgr = cv2.cvtColor(img_opencv_rgb, cv2.COLOR_RGB2BGR)
    #img_opencv_rgb = np.rot90(img_opencv_rgb,1,(1,0))

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img_opencv_bgr = img_opencv_bgr.byteswap().newbyteorder()

    return img_opencv_rgb

  def cv2_to_imgmsg(self, cv_image):
    img_msg = Image()

    scale_percent = self.scale # percent of original size
    width = int(cv_image.shape[1] * scale_percent / 100)
    height = int(cv_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    cv_image = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
    cv_image = np.rot90(cv_image,1,(1,0))

    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "rgb8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

    
def get_model():
  #model = torch.hub.load('/home/docker_share/fvd/yolo/yolov5', 
  #                      'custom', 
  #                      path='/home/docker_share/fvd/yolo/saved_models/best.pt', 
  #                      source='local')
  
  '''
  model = torch.hub.load('/home/bob/yolo/yolov5', 
                        'custom', 
                        path='/home/bob/yolo/saved_models/best.pt', 
                        source='local')
  '''

  '''
  model = torch.hub.load('/home/bob/yolo/yolov5', 
                        'custom', 
                        path='/home/bob/yolo/saved_models/best_fp_data.pt', 
                        source='local')
  '''

  model = torch.hub.load('/home/bob/yolo/yolov5', 
                        'custom', 
                        path='/home/bob/yolo/saved_models/exp3_best.pt', 
                        source='local')

  return model


def main(args):

  model = get_model()
  model.eval()
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print(device)
  model.to(device)

  transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

  rospy.init_node('image_converter', anonymous=True)

  ic = image_converter(model, transform, device)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  main(sys.argv)
