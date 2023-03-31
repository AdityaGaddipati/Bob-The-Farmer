#include <ros/ros.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <visualization_msgs/MarkerArray.h>

#include <std_msgs/String.h>
#include <string>

#include <tf/tf.h>

#include <cmath>

ros::Publisher pub;

float cluster_x=0, cluster_y=0, cluster_z=0; 
std_msgs::String target_frame;
bool received_pt = false;

tf2_ros::Buffer tfBuffer;

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud){

    if(received_pt!= true) return;

    target_frame.data="odom";

    geometry_msgs::TransformStamped transformStamped;
    try{
      transformStamped = tfBuffer.lookupTransform("head_camera_color_optical_frame", target_frame.data, ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      return;
    }


    tf::Quaternion q(
        transformStamped.transform.rotation.x,
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z,
        transformStamped.transform.rotation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    
    pcl::PCLPointCloud2 cloud_filtered;
    pcl::CropBox<pcl::PCLPointCloud2> crop;
    crop.setInputCloud(cloud);

    Eigen::Vector3f box_center = Eigen::Vector3f(cluster_x, cluster_y, cluster_z);
    Eigen::Vector3f box_rotation = Eigen::Vector3f(roll, pitch, yaw);
    // Eigen::Vector3f box_rotation = Eigen::Vector3f(-roll, -pitch, -yaw);
    // Eigen::Vector4f min_point = Eigen::Vector4f(-0.1, -0.05, -0.02, 1.0f);
    // Eigen::Vector4f max_point = Eigen::Vector4f(0.2, 0.05, 0.02, 1.0f);
    Eigen::Vector4f min_point = Eigen::Vector4f(-0.05, -0.05, -0.2, 1.0f);
    Eigen::Vector4f max_point = Eigen::Vector4f(0.05, 0.05, 0.2, 1.0f);


    // Weird case
    // Eigen::Vector4f min_point = Eigen::Vector4f(-0.4, -0.1, -0.01, 1.0f);
    // Eigen::Vector4f max_point = Eigen::Vector4f(0.1, 0.1, 0.01, 1.0f);

    crop.setTranslation(box_center);
    crop.setRotation(box_rotation);

    // Eigen::Vector4f min_point = Eigen::Vector4f(cluster_x-0.25, cluster_y-0.05, cluster_z-0.02, 1.0f);
    // Eigen::Vector4f max_point = Eigen::Vector4f(cluster_x+0.2, cluster_y+0.05, cluster_z+0.02, 1.0f);

    crop.setMin(min_point);
    crop.setMax(max_point);

    crop.filter(cloud_filtered);

    // Publish the data
    pub.publish (cloud_filtered);

    
    std_msgs::String msg;
    std::stringstream ss;
    // ss << cloud.height << " " << cloud.width << " " << cloud.header.frame_id;
    ss << roll*(180/3.14) << " " << pitch*(180/3.14) << " " << yaw*(180/3.14);
    msg.data = ss.str();
    ROS_INFO("%s", msg.data.c_str());

    received_pt = false;
    
}


void cluster_cb(const geometry_msgs::PoseStamped &poseStamped){
  
    target_frame.data = poseStamped.header.frame_id;
    cluster_x = poseStamped.pose.position.x;
    cluster_y = poseStamped.pose.position.y;
    cluster_z = poseStamped.pose.position.z;
    
    std_msgs::String msg;

    std::stringstream ss;
    ss << cluster_x << " " << cluster_y << " " << cluster_z;
    msg.data = ss.str();

    ROS_INFO("%s", msg.data.c_str());

    received_pt = true;
}


void cluster_cb_test(const visualization_msgs::MarkerArray &marker){
  
  cluster_x = marker.markers[0].pose.position.x;
  cluster_y = marker.markers[0].pose.position.y;
  cluster_z = marker.markers[0].pose.position.z;

  std_msgs::String msg;

  std::stringstream ss;
  ss << cluster_x << " " << cluster_y << " " << cluster_z;
  msg.data = ss.str();

  // ROS_INFO("%s", msg.data.c_str());

  received_pt = true;

}


void dummy_cb (const pcl::PCLPointCloud2ConstPtr& cloud){

  
    target_frame.data="odom";

    geometry_msgs::TransformStamped transformStamped;
    try{
      transformStamped = tfBuffer.lookupTransform("head_camera_color_optical_frame", target_frame.data, ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      return;
    }


    tf::Quaternion q(
        transformStamped.transform.rotation.x,
        transformStamped.transform.rotation.y,
        transformStamped.transform.rotation.z,
        transformStamped.transform.rotation.w);
    tf::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // double roll=90*(M_PI/180), pitch=0*(M_PI/180), yaw=90*(M_PI/180);

    cluster_x = 0.5;
    cluster_y = 0.5;
    cluster_z = 0.5;

    
    pcl::PCLPointCloud2 cloud_filtered;
    pcl::CropBox<pcl::PCLPointCloud2> crop;
    crop.setInputCloud(cloud);

    Eigen::Vector3f box_center = Eigen::Vector3f(cluster_x, cluster_y, cluster_z);
    Eigen::Vector3f box_rotation = Eigen::Vector3f(roll, pitch, yaw);
    Eigen::Vector4f min_point = Eigen::Vector4f(-0.1, -0.3, -0.5, 1.0f);
    Eigen::Vector4f max_point = Eigen::Vector4f(0.1, 0.3, 0.5, 1.0f);


    crop.setTranslation(box_center);
    crop.setRotation(box_rotation);
    crop.setMin(min_point);
    crop.setMax(max_point);

    crop.filter(cloud_filtered);

    // Publish the data
    pub.publish (cloud_filtered);

    
    std_msgs::String msg;
    std::stringstream ss;
    // ss << cloud.height << " " << cloud.width << " " << cloud.header.frame_id;
    ss << roll*(180/3.14) << " " << pitch*(180/3.14) << " " << yaw*(180/3.14);
    msg.data = ss.str();
    ROS_INFO("%s", msg.data.c_str());


    received_pt = false;

    
}


int main (int argc, char** argv){
    // Initialize ROS
    ros::init (argc, argv, "my_pcl_extractor");
    ros::NodeHandle nh;

    tf2_ros::TransformListener tfListener(tfBuffer);

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub1 = nh.subscribe ("head_camera/depth/color/points", 1, cloud_cb);
    ros::Subscriber sub2 = nh.subscribe ("exctraction_pt", 1, cluster_cb);

    // ros::Subscriber sub3 = nh.subscribe ("cluster_loc1", 1, cluster_cb_test);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<pcl::PCLPointCloud2> ("extracted_cloud", 1);

    // ros::Subscriber sub4 = nh.subscribe ("point_cloud2", 1, dummy_cb);


    ros::spin ();
}
