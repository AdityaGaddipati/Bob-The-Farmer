#include <ros/ros.h>
// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>

ros::Publisher pub;

void 
cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud)
{
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
  Eigen::Vector4f min_point = Eigen::Vector4f(-0.1f, -0.1f, 0.0f, 1.0f);
  Eigen::Vector4f max_point = Eigen::Vector4f(0.5f, 0.1f, 0.6f, 1.0f);
  crop.setMin(min_point);
  crop.setMax(max_point);
  crop.filter(cloud_filtered);

  // Publish the data
  pub.publish (cloud_filtered);
}

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("head_camera/depth/color/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<pcl::PCLPointCloud2> ("output", 1);
  std::cout << "Here";
  // Spin
  ros::spin ();
}
