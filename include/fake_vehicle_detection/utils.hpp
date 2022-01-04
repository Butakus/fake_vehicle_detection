#ifndef FAKE_VEHICLE_DETECTION__UTILS_HPP_
#define FAKE_VEHICLE_DETECTION__UTILS_HPP_

#include <cmath>

#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
// Remove "-Wpedantic with tf2/utils.h to avoid warnings about extra ';'"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <tf2/utils.h>
#pragma GCC diagnostic pop

namespace fake_vehicle_detection
{

// Creates and returns a quaternion msg from a given yaw angle
template<typename T>
geometry_msgs::msg::Quaternion quaternion_msg_from_yaw(T yaw)
{
    geometry_msgs::msg::Quaternion q;
    tf2::Quaternion tf_q;

    tf_q.setRPY(0.0, 0.0, yaw);

    q.x = tf_q.x();
    q.y = tf_q.y();
    q.z = tf_q.z();
    q.w = tf_q.w();

    return q;
}

// Extracts yaw angle from quaternion msg
template<typename T>
T yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    return tf2::getYaw(tf_q);
}

// Converts degrees to radians in compile time
template<typename T>
constexpr double deg_to_rad(const T& deg) {return (M_PI * deg / 180.0);}

// Smooth error function f(x) = x / (k + abs(x)), where k is the smoothness parameter
// template<typename T>
// T smooth(T x, T smoothness);
double smooth(const double& x, const double& smoothness)
{
    return (x / (smoothness + std::abs(x)));
}

// Angle normalization to [0-360] range
template<typename T>
T norm_angle(const T& angle)
{
    return angle < 0 ? angle + deg_to_rad(360.0) : angle;
}

// Distance between poses
double compute_distance(geometry_msgs::msg::Pose p1, geometry_msgs::msg::Pose p2)
{
    double dx = p2.position.x - p1.position.x;
    double dy = p2.position.y - p1.position.y;
    double dz = p2.position.z - p1.position.z;
    double distance = std::hypot(dx, dy, dz);
    return distance;
}

geometry_msgs::msg::Pose relative_transform(geometry_msgs::msg::Pose p1, geometry_msgs::msg::Pose p2)
{
    tf2::Transform p1_tf;
    tf2::fromMsg(p1, p1_tf);
    tf2::Transform p2_tf;
    tf2::fromMsg(p2, p2_tf);

    tf2::Transform transform_tf = p1_tf.inverseTimes(p2_tf);

    geometry_msgs::msg::Pose transform_pose;
    tf2::toMsg(transform_tf, transform_pose);

    return transform_pose;
}

} // Namespace fake_vehicle_detection

#endif // FAKE_VEHICLE_DETECTION__UTILS_HPP_
