#include <ros/ros.h>
#include <plan_env/grid_map.h>
#include <humanoid_navigation/RawVoxelMap.h>
#include <array>

int main(int argc, char** argv) {
  ros::init(argc, argv, "humanoid_static_grid_map");
  ros::NodeHandle nh("~"), public_nh;
  std::string scene;
  nh.param<std::string>("scene", scene, "compound");
  if (scene != "open" && scene != "narrow" && scene != "low" && scene != "compound" && scene != "blocked") {
    ROS_FATAL("Unknown scene"); return 1;
  }
  double resolution;
  nh.param("map_resolution", resolution, 0.05);
  if (!std::isfinite(resolution) || resolution < .01 || resolution > .2) {
    ROS_FATAL("map_resolution must be in [0.01, 0.2]"); return 1;
  }
  GridMap map;
  map.initStaticMap(Eigen::Vector3d(-3,-2,0), Eigen::Vector3d(6,4,2), resolution);
  double width = (scene == "blocked") ? .30 : ((scene == "narrow" || scene == "compound") ? .46 : 1.2);
  double ceiling = (scene == "low" || scene == "compound") ? 1.22 : 2.;
  std::vector<std::array<double,6>> blocks;
  if (scene != "open") {
    blocks.push_back({{-.3,-2,0,.3,-width/2,2}});
    blocks.push_back({{-.3,width/2,0,.3,2,2}});
    if (ceiling < 2) blocks.push_back({{-.3,-width/2,ceiling,.3,width/2,2}});
  }
  humanoid_navigation::RawVoxelMap message;
  message.header.frame_id = "world";
  message.header.stamp = ros::Time::now();
  message.resolution = resolution;
  message.origin[0] = -3; message.origin[1] = -2; message.origin[2] = 0;
  for (int i=0;i<3;++i) message.shape[i] = std::ceil((i==0 ? 6. : (i==1 ? 4. : 2.))/resolution);
  // Rasterize conservatively using half-open voxel ranges, then read raw storage.
  for (const auto& box : blocks) {
    Eigen::Vector3i low, high;
    for (int i=0;i<3;++i) {
      low(i) = std::max(0, static_cast<int>(std::floor((box[i]-message.origin[i])/resolution+1e-9)));
      high(i) = std::min(static_cast<int>(message.shape[i]), static_cast<int>(std::ceil((box[i+3]-message.origin[i])/resolution-1e-9)));
    }
    for (int x=low(0);x<high(0);++x) for (int y=low(1);y<high(1);++y) for (int z=low(2);z<high(2);++z)
      map.setRawOccupancy(Eigen::Vector3i(x,y,z), 1);
  }
  for (unsigned x=0;x<message.shape[0];++x) for (unsigned y=0;y<message.shape[1];++y) for (unsigned z=0;z<message.shape[2];++z)
    message.data.push_back(map.getRawOccupancy(Eigen::Vector3i(x,y,z)));
  ros::Publisher pub = public_nh.advertise<humanoid_navigation::RawVoxelMap>("raw_map", 1, true);
  pub.publish(message);
  ros::spin();
}
