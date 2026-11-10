# Uninflated plan_env snapshot. x-major, then y, then z (z fastest).
# Each data entry: -1 unknown, 0 free, 1 occupied. Origin is a voxel corner.
std_msgs/Header header
float64 resolution
float64[3] origin
uint32[3] shape
int8[] data
