#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
import torchvision 
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    img_opencv_rgb = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)
    
    #img_opencv_bgr = cv2.cvtColor(img_opencv_rgb, cv2.COLOR_RGB2BGR)
    #img_opencv_rgb = np.rot90(img_opencv_rgb,1,(1,0))

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img_opencv_bgr = img_opencv_bgr.byteswap().newbyteorder()

    return img_opencv_rgb

def cv2_to_imgmsg(cv_image):
    img_msg = Image()

    scale_percent = 50 # percent of original size
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


class image_converter:

  def __init__(self, model, transform, device):
    self.image_pub = rospy.Publisher("/inference",Image, queue_size=10)

    self.bridge = CvBridge()
    # self.image_sub = rospy.Subscriber("camera/color/image_raw",Image,self.callback, queue_size=30)
    self.image_sub = rospy.Subscriber("head_camera/color/image_raw",Image,self.callback, queue_size=30)

    self.model = model
    self.transforms = transform
    self.device = device

  def callback(self,data):
    #print(data.header.stamp.secs)
    
    try:
      #cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
      cv_image = imgmsg_to_cv2(data)
    except CvBridgeError as e:
      print(e)

    bbox,scores = self.run_inference(np.asarray(cv_image))
    
    
    #print(len(bbox),len(scores))

    for i in range(len(bbox)):
      # print(bbox)
      if(scores[i]>0.7):
        bb = bbox[i]
        cv2.rectangle(cv_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 5)

    try:
      #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "rgb8"))
      self.image_pub.publish(cv2_to_imgmsg(cv_image))
    except CvBridgeError as e:
      print(e)
    

  def run_inference(self, img):
    #img = self.transforms(img)
    print(img.shape)
    
    with torch.no_grad():
      results = self.model(img)
      results_np = results.xyxy[0].detach().cpu().numpy()

    print("*****")
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

    print(bbox)
    print("\n")
    print(scores)
    print("\n")

    return bbox,scores
    

def get_model(num_classes):
  model = torch.hub.load('/home/bob/yolo/yolov5','custom', path='/home/bob/yolo/saved_models/best.pt', source='local')  
  # model = torch.hub.load('./yolov5','custom', path='/home/bob/yolo/saved_models/best.pt')  
  

  #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # in_features = model.roi_heads.box_predictor.cls_score.in_features
  #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model


def main(args):

  model = get_model(num_classes=2)
  #model.load_state_dict(torch.load("saved_models/checkpoint-FasterRCNN_Tomato-epoch50.pth"))
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
