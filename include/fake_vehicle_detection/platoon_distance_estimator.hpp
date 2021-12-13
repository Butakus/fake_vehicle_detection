#ifndef FAKE_VEHICLE_DETECTION__PLATOON_DISTANCE_ESTIMATOR_HPP_
#define FAKE_VEHICLE_DETECTION__PLATOON_DISTANCE_ESTIMATOR_HPP_

#include <random>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <loco_framework/msg/platoon_detection.hpp>

namespace fake_vehicle_detection
{

class PlatoonDistanceEstimator : public rclcpp::Node
{
public:
    using Pose = geometry_msgs::msg::PoseStamped;
    using PlatoonDetection = loco_framework::msg::PlatoonDetection;

    // Smartpointer typedef
    typedef std::shared_ptr<PlatoonDistanceEstimator> SharedPtr;
    typedef std::unique_ptr<PlatoonDistanceEstimator> UniquePtr;

    PlatoonDistanceEstimator();
    PlatoonDistanceEstimator(const rclcpp::NodeOptions& options);
    virtual ~PlatoonDistanceEstimator();

protected:
    // Parameters
    double distance_stddev_;
    double bad_measurement_probability_;
    double rate_;
    OnSetParametersCallbackHandle::SharedPtr set_param_callback_handler_;

    // State
    Pose::SharedPtr ego_pose_;
    Pose::SharedPtr leader_pose_;
    Pose::SharedPtr follower_pose_;
    std::mutex ego_mutex_;
    std::mutex leader_mutex_;
    std::mutex follower_mutex_;

    // Threads
    bool running_ = false;
    std::thread executor_thread_;

    // Random engine
    std::random_device rd_;
    std::mt19937 random_generator_;
    // Random distributions
    std::uniform_real_distribution<> uniform_distribution_;
    std::exponential_distribution<> exponential_distribution_;

    // Publishers
    rclcpp::Publisher<PlatoonDetection>::SharedPtr leader_detection_pub_;
    rclcpp::Publisher<PlatoonDetection>::SharedPtr follower_detection_pub_;

    // Subscribers
    rclcpp::Subscription<Pose>::SharedPtr ego_pose_sub_;
    rclcpp::Subscription<Pose>::SharedPtr leader_pose_sub_;
    rclcpp::Subscription<Pose>::SharedPtr follower_pose_sub_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    void init();
    void run();

    void publish_leader_detection();
    void publish_follower_detection();

    // Compute the measurement noise. Returns a tuple with the noise addition and the stddev
    std::tuple<double, double> compute_noise();

    // Param callback
    rcl_interfaces::msg::SetParametersResult set_param_callback(const std::vector<rclcpp::Parameter>& params);

    // Callbacks
    void ego_pose_callback(const Pose::SharedPtr ego_pose_msg);
    void leader_pose_callback(const Pose::SharedPtr leader_pose_msg);
    void follower_pose_callback(const Pose::SharedPtr follower_pose_msg);
};


}  // namespace fake_vehicle_detection

#endif  // FAKE_VEHICLE_DETECTION__PLATOON_DISTANCE_ESTIMATOR_HPP_
