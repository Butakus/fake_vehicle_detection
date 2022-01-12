#ifndef FAKE_VEHICLE_DETECTION__LIDAR_VEHICLE_DETECTION_HPP_
#define FAKE_VEHICLE_DETECTION__LIDAR_VEHICLE_DETECTION_HPP_

#include <random>
#include <mutex>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <loco_framework/msg/detection_array.hpp>

#include <fake_vehicle_detection/utils.hpp>
#include <fake_vehicle_detection/mvn.hpp>

namespace fake_vehicle_detection
{

class LidarVehicleDetection : public rclcpp::Node
{
public:
    using Pose = geometry_msgs::msg::PoseStamped;
    using Detection = geometry_msgs::msg::PoseWithCovarianceStamped;
    using DetectionArray = loco_framework::msg::DetectionArray;

    // Smartpointer typedef
    typedef std::shared_ptr<LidarVehicleDetection> SharedPtr;
    typedef std::unique_ptr<LidarVehicleDetection> UniquePtr;

    LidarVehicleDetection();
    LidarVehicleDetection(const rclcpp::NodeOptions& options);
    virtual ~LidarVehicleDetection();

protected:
    // Parameters
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
    rclcpp::Publisher<DetectionArray>::SharedPtr detections_pub_;

    // Subscribers
    rclcpp::Subscription<Pose>::SharedPtr ego_pose_sub_;
    std::vector<rclcpp::Subscription<Pose>::SharedPtr> pose_subs_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    void init();
    void run();

    // Compute the detection from ego pose to target pose i
    Detection compute_detection(const size_t i);
    // Get the noise covariance from the measurement model
    Eigen::Matrix<double, 2, 2> get_noise_covariance(const double distance, const double angle);
    // Compute the measurement noise. Returns the noise covariance
    Eigen::Vector2d generate_noise(const Eigen::Matrix<double, 2, 2>& noise_covariance);

    // Param callback
    rcl_interfaces::msg::SetParametersResult set_param_callback(const std::vector<rclcpp::Parameter>& params);

    // Callbacks
    void ego_pose_callback(const Pose::SharedPtr ego_pose_msg);
};


}  // namespace fake_vehicle_detection

#endif  // FAKE_VEHICLE_DETECTION__LIDAR_VEHICLE_DETECTION_HPP_
