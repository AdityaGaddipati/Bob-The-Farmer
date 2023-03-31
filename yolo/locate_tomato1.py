#!/usr/bin/env python3
from __future__ import print_function

import sys
import math
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import message_filters

from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Quaternion, Twist, Vector3, PoseArray

#Visualiztion stuff
from visualization_msgs.msg import MarkerArray, Marker

class image_converter:

  def __init__(self, model, transform, device):
    self.image_pub = rospy.Publisher("/inference",Image, queue_size=10)
    self.depth_pub = rospy.Publisher("/depth_inference",Image, queue_size=10)

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
    self.thr = 80
    # Vis image scale
    self.scale = 50
    self.intrinsics = None
    self.cv_image = None
    self.fx = None
    self.cx = None
    self.fy = None
    self.cy = None


  def calc_centroid(self, bb):
    # print(bb)
    cx = (bb[0]+bb[2])/2
    cy = (bb[1]+bb[3])/2
    # print(cx, cy)
    # print(self.red[0,0])
    # print(self.red[int(cx-3):int(cx+3),int(cy-3):int(cy+3)])
    if self.red[int(cy),int(cx)]==0:
      return -1, -1
    #print("centroid: ", cx, cy)
    #cv2.circle(self.red, (int(cy),int(cx)), 1, (0, 255, 0), 3)
    #cv2.imshow("Red", self.red)
    #cv2.waitKey(1000)
    return (int(cx), int(cy))

  def calc_big_bb(self, bb):
    big_bb = np.zeros((1,4))
    big_bb[:, 0] = np.min(bb[:,0])
    big_bb[:, 1] = np.min(bb[:,1])
    big_bb[:, 2] = np.max(bb[:,2])
    big_bb[:, 3] = np.max(bb[:,3])
    return big_bb
  
  def bbox_cutoff(self, bbox, scores):
    scored_bbox = None
    for i in range(len(bbox)):
      if(scores[i]>self.conf_thresh):
        if(scored_bbox is None):
          scored_bbox = np.asarray(bbox[i]).reshape(1,4)
        else:
          scored_bbox = np.vstack((scored_bbox, bbox[i]))
        bb = bbox[i]
        #cv2.rectangle(self.cv_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 2)

    return scored_bbox

  def cluster(self, bbox):
    #print("*****")
    #print(bbox)
    bbox = bbox[np.argsort(bbox[:, 0])]
    #print(bbox)
    bigbb = []
    bigbb_list = []

    Thr = self.thr
    
    # print(bbox.shape)
    rectsUsed = np.zeros(bbox.shape[0])
    # Iterate all initial bounding rects
    for bbIdx, bbVal in enumerate(bbox):
        bb_list = []
        if (rectsUsed[bbIdx] == False):
            # Initialize current rect
            currxMin = bbVal[0]
            currxMax = bbVal[2]
            curryMin = bbVal[1]
            curryMax = bbVal[3]
            # This bounding rect is used
            rectsUsed[bbIdx] = True
            bb_list.append(bbIdx)
            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(bbox[(bbIdx+1):], start = (bbIdx+1)):
                if (rectsUsed[subIdx] == True):
                  continue
                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[3]

                if (candyMax > curryMin-Thr and candyMin < curryMin+Thr  and candxMin < currxMax + Thr and candxMax > currxMax-Thr) or (candyMin < curryMax+Thr and candyMax>curryMax-Thr and candxMin < currxMax + Thr and candxMax>currxMax-Thr):
                      currxMin = min(currxMin, candxMin)
                      currxMax = max(currxMax, candxMax)
                      curryMin = min(curryMin, candyMin)
                      curryMax = max(curryMax, candyMax)
                      rectsUsed[subIdx] = True
                      bb_list.append(subIdx)

                else:
                    break
            # No more merge candidates possible, accept current rect
            bigbb.append([currxMin, curryMin, currxMax, curryMax])
            bigbb_list.append(bb_list)
    for rect in bigbb:
        cv2.rectangle(self.cv_image, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0), 2)

    return bigbb, bigbb_list 
  
  def calc_depth(self, bigbb_list, bbox):
      camera_matrix = np.array([[910.7079467773438, 0.0, 634.5316772460938], [0.0, 910.6213989257812, 355.40097045898], [0.0, 0.0, 1.0]]) 

      depth_data = np.frombuffer(self.depth_image.data, dtype=np.uint16).reshape(self.depth_image.height, self.depth_image.width, -1)
      # print("size of image", self.cv_image.shape, depth_data.shape)
      
      coordinate_list = []
      for idx, bb_list in enumerate(bigbb_list):
        # xsum1 = []
        # ysum1 = []
        # zsum1 = []
        xsum2 = []
        ysum2 = []
        zsum2 = []
        for bb in bb_list:
          # print(len(bb_list))
          c = self.calc_centroid(bbox[bb])
          if(c[0]==-1 and c[1]==-1):
            continue

          Z = depth_data[c[1], c[0]]
          if(Z==0):
            continue

          # result1 = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [720 - c[0], c[1]], Z)
          # xsum1.append(result1[0])
          # ysum1.append(result1[1])
          # zsum1.append(Z)
          
          zsum2.append(Z)
          xsum2.append(((c[0] - self.cx) / self.fx) * Z)
          ysum2.append(((c[1] - self.cy) / self.fy) * Z)
        
        coordinate_list.append((np.mean(xsum2)/1000, np.mean(ysum2)/1000, np.mean(zsum2)/1000))

      return coordinate_list

  def detect_color(self, image):
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    '''Red'''
    # Range for lower red
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
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
    '''
    try:
        if self.intrinsics:
            return
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]
    except CvBridgeError as e:
        print(e)
        return
    '''

  def callback(self,data, depth_image):
    print(data.header.frame_id, depth_image.header.frame_id)
    # print(data.height, data.width)
    try:
      #self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
      self.cv_image = self.imgmsg_to_cv2(data)
      self.color_image = self.cv_image.copy()
      self.depth_image = depth_image

      # print("size of cv image", self.cv_image.shape)
	
      preds = self.run_inference(np.asarray(self.cv_image))

      bbox = preds['boxes'].cpu().detach().numpy()
      scores = preds['scores'].cpu().detach().numpy()

      bbox = self.bbox_cutoff(bbox, scores)
      if(bbox is None):
        print("No detections")
        return

      self.red = self.detect_color(self.color_image)
      
      bigbb, bigbb_list = self.cluster(bbox)
      print("bigbb: ", bigbb)
      print("bigbb_list: ", bigbb_list)

      marker_array = MarkerArray()
      pose_array = PoseArray()
      pose_array.header = data.header

      coordinate_list = self.calc_depth(bigbb_list, bbox)
      for idx, coordinates in enumerate(coordinate_list):
        print(idx, " Coordinates:", coordinates[0], coordinates[1], coordinates[2])
        if math.isnan(coordinates[2]):
            continue
        marker = Marker()
        marker.header.frame_id = data.header.frame_id
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
      
    except Exception as e:
      print(e)


  def run_inference(self, img):
    img = self.transforms(img)
    with torch.no_grad():
      preds = self.model(img.unsqueeze(0).to(self.device))[0]

    return preds

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

    
def get_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model


def main(args):

  #print(torch.__version__)
  #print(torchvision.__version__)

  model = get_model(num_classes=2)
  model.load_state_dict(torch.load("saved_models/checkpoint-FasterRCNN_Tomato-epoch50.pth"))
  #model.load_state_dict(torch.load("saved_models/checkpoint-FasterRCNN_Tomato-epoch249.pth"))
  
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
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
