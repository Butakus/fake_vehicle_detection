#ifndef FAKE_VEHICLE_DETECTION__PLATOON_MULTI_DISTANCE_ESTIMATOR_HPP_
#define FAKE_VEHICLE_DETECTION__PLATOON_MULTI_DISTANCE_ESTIMATOR_HPP_

#include <random>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <loco_framework/msg/platoon_detection_array.hpp>

namespace fake_vehicle_detection
{

class PlatoonMultiDistanceEstimator : public rclcpp::Node
{
public:
    using Pose = geometry_msgs::msg::PoseStamped;
    using PlatoonDetection = loco_framework::msg::PlatoonDetection;
    using PlatoonDetectionArray = loco_framework::msg::PlatoonDetectionArray;

    // Smartpointer typedef
    typedef std::shared_ptr<PlatoonMultiDistanceEstimator> SharedPtr;
    typedef std::unique_ptr<PlatoonMultiDistanceEstimator> UniquePtr;

    PlatoonMultiDistanceEstimator();
    PlatoonMultiDistanceEstimator(const rclcpp::NodeOptions& options);
    virtual ~PlatoonMultiDistanceEstimator();

protected:
    // Parameters
    double distance_stddev_;
    double bad_measurement_probability_;
    double rate_;
    OnSetParametersCallbackHandle::SharedPtr set_param_callback_handler_;

    // State
    size_t number_of_vehicles_;
    Pose::SharedPtr ego_pose_;
    std::vector<Pose> poses_;
    std::vector<bool> poses_received_;
    std::mutex ego_mutex_;
    std::vector<std::mutex> pose_mutexes_;

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
    rclcpp::Publisher<PlatoonDetectionArray>::SharedPtr detections_pub_;

    // Subscribers
    rclcpp::Subscription<Pose>::SharedPtr ego_pose_sub_;
    std::vector<rclcpp::Subscription<Pose>::SharedPtr> pose_subs_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    void init();
    void run();

    // Compute the distance fromego pose to target pose i
    PlatoonDetection compute_distance(const size_t i);
    // Compute the measurement noise. Returns a tuple with the noise addition and the stddev
    std::tuple<double, double> compute_noise();

    // Param callback
    rcl_interfaces::msg::SetParametersResult set_param_callback(const std::vector<rclcpp::Parameter>& params);

    // Callbacks
    void ego_pose_callback(const Pose::SharedPtr ego_pose_msg);
};


}  // namespace fake_vehicle_detection

#endif  // FAKE_VEHICLE_DETECTION__PLATOON_MULTI_DISTANCE_ESTIMATOR_HPP_
