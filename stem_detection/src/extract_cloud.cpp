#include <ros/ros.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <std_msgs/String.h>

ros::Publisher pub;

float cluster_x=0, cluster_y=0, cluster_z=0; 

void cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud){
  pcl::PCLPointCloud2 cloud_filtered;

  // Perform the actual filtering
  /*
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.1, 0.1, 0.1);
  sor.filter (cloud_filtered);
  */

  pcl::CropBox<pcl::PCLPointCloud2> crop;
  crop.setInputCloud(cloud);

  //Eigen::Vector4f min_point = Eigen::Vector4f(-0.1f, -0.1f, 0.0f, 1.0f);
  //Eigen::Vector4f max_point = Eigen::Vector4f(0.5f, 0.1f, 0.6f, 1.0f);

  //Eigen::Vector4f min_point = Eigen::Vector4f(-1.0f, -1.0f, -1.0f, 1.0f);
  //Eigen::Vector4f max_point = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

  Eigen::Vector4f min_point = Eigen::Vector4f(cluster_x-0.1, cluster_y-0.05, cluster_z-0.02, 1.0f);
  Eigen::Vector4f max_point = Eigen::Vector4f(cluster_x+0.2, cluster_y+0.05, cluster_z+0.02, 1.0f);

  crop.setMin(min_point);
  crop.setMax(max_point);
  crop.filter(cloud_filtered);

  // Publish the data
  pub.publish (cloud_filtered);

  /*
  std_msgs::String msg;
  std::stringstream ss;
  ss << cloud.height << " " << cloud.width << " " << cloud.header.frame_id;
  msg.data = ss.str();
  */

}


void cluster_cb(const visualization_msgs::MarkerArray &marker){
  
  cluster_x = marker.markers[0].pose.position.x;
  cluster_y = marker.markers[0].pose.position.y;
  cluster_z = marker.markers[0].pose.position.z;

  std_msgs::String msg;

  std::stringstream ss;
  ss << cluster_x << " " << cluster_y << " " << cluster_z;
  msg.data = ss.str();

  ROS_INFO("%s", msg.data.c_str());

}


int main (int argc, char** argv){
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub1 = nh.subscribe ("head_camera/depth/color/points", 1, cloud_cb);
  ros::Subscriber sub2 = nh.subscribe ("cluster_loc1", 1, cluster_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<pcl::PCLPointCloud2> ("output", 1);

  // Spin
  ros::spin ();
}
