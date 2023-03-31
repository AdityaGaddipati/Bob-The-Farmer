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

    #if img_msg.encoding != "bgr8":
    #    rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    
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
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback, queue_size=30)

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

    preds = self.run_inference(np.asarray(cv_image))
    # print(preds)
    
    '''
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
    lower1 = np.array([20, 100, 100])
    upper1 = np.array([40, 255, 255])
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255]) 
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    upper_mask = cv2.inRange(hsv, lower2, upper2) 
    red_mask = lower_mask + upper_mask
    red = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)
    cv_image = red
    '''
    '''
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
    #low_red = np.array([161, 155, 84])
    #high_red = np.array([179, 255, 255])
    low_red = np.array([30,150,50])
    high_red = np.array([255,255,180])
    red_mask = cv2.inRange(hsv, low_red, high_red)
    red = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)
    cv_image = red
    '''

    bbox = preds['boxes'].cpu().detach().numpy()
    scores = preds['scores'].cpu().detach().numpy()


    for i in range(len(bbox)):
      # print(bbox)
      if(scores[i]>0.7):
        bb = bbox[i]
        cv2.rectangle(cv_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 5)

    try:
      #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "rgb8"))
      self.image_pub.publish(cv2_to_imgmsg(cv_image))
    except CvBridgeError as e:
      print(e)


  def run_inference(self, img):
    img = self.transforms(img)
    with torch.no_grad():
      preds = self.model(img.unsqueeze(0).to(self.device))[0]
    print(preds)
    return preds
    

def get_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model


def main(args):

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
