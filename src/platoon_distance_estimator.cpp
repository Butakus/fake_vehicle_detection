#include "fake_vehicle_detection/platoon_distance_estimator.hpp"

namespace fake_vehicle_detection
{

PlatoonDistanceEstimator::PlatoonDistanceEstimator() :
    rclcpp::Node("platoon_distance_estimator"),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

PlatoonDistanceEstimator::PlatoonDistanceEstimator(const rclcpp::NodeOptions& options) :
    rclcpp::Node("platoon_distance_estimator", options),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

PlatoonDistanceEstimator::~PlatoonDistanceEstimator()
{
    this->running_ = false;
    if (this->executor_thread_.joinable())
    {
        this->executor_thread_.join();
    }
}

void PlatoonDistanceEstimator::init()
{
    using std::placeholders::_1;
    // Initialize and declare parameters
    this->distance_stddev_ = this->declare_parameter("distance_stddev", 0.05);
    this->bad_measurement_probability_ = this->declare_parameter("bad_measurement_probability", 0.05);
    this->rate_ = this->declare_parameter("rate", 20.0);

    // Set callback to handle parameter setting
    this->set_param_callback_handler_ = this->add_on_set_parameters_callback(
                                                std::bind(&PlatoonDistanceEstimator::set_param_callback, this, _1));

    // Initialize random distributions
    this->uniform_distribution_ = std::uniform_real_distribution<>(0.0, 1.0);
    this->exponential_distribution_ = std::exponential_distribution<>(2.5);

    // Publishers
    this->leader_detection_pub_ =
        this->create_publisher<PlatoonDetection>("~/leader_distance", rclcpp::SensorDataQoS());
    this->follower_detection_pub_ =
        this->create_publisher<PlatoonDetection>("~/follower_distance", rclcpp::SensorDataQoS());

    // Create a reentrant callback_group for callbacks, so they can be called concurrently
    this->callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    auto sub_options = rclcpp::SubscriptionOptions();
    sub_options.callback_group = this->callback_group_;

    // Subscribers
    this->ego_pose_sub_ = this->create_subscription<Pose>(
        "pose", rclcpp::SensorDataQoS(),
        std::bind(&PlatoonDistanceEstimator::ego_pose_callback, this, _1), sub_options);
    this->leader_pose_sub_ = this->create_subscription<Pose>(
        "leader_pose", rclcpp::SensorDataQoS(),
        std::bind(&PlatoonDistanceEstimator::leader_pose_callback, this, _1), sub_options);
    this->follower_pose_sub_ = this->create_subscription<Pose>(
        "follower_pose", rclcpp::SensorDataQoS(),
        std::bind(&PlatoonDistanceEstimator::follower_pose_callback, this, _1), sub_options);

    // Start the main execution thread to update and publish the state
    this->executor_thread_ = std::thread(&PlatoonDistanceEstimator::run, this);
}

void PlatoonDistanceEstimator::run()
{
    this->running_ = true;

    // First, wait until ego pose is received
    rclcpp::Rate rate(this->rate_);
    while (rclcpp::ok() && this->running_ && this->ego_pose_ == nullptr)
    {
        rate.sleep();
    }

    while (rclcpp::ok() && this->running_)
    {
        if (this->leader_pose_ != nullptr)
        {
            // Process and publish leader distance
            this->publish_leader_detection();
        }

        if (this->follower_pose_ != nullptr)
        {
            // Process and publish follower distance
            this->publish_follower_detection();
        }

        rate.sleep();
    }
}

rcl_interfaces::msg::SetParametersResult PlatoonDistanceEstimator::set_param_callback(const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & param : params)
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Set param " << param.get_name() << ": " << param.value_to_string());
        if (param.get_name() == "distance_stddev")
        {
            // Lock only ego mutex, this should be enough to prevent actions where the gauss distribution is used
            std::lock_guard<std::mutex> lock(this->ego_mutex_);
            this->distance_stddev_ = param.as_double();
        }
        else if (param.get_name() == "bad_measurement_probability")
        {
            // Lock only ego mutex, this should be enough to prevent actions where the gauss distribution is used
            std::lock_guard<std::mutex> lock(this->ego_mutex_);
            this->bad_measurement_probability_ = param.as_double();
        }
        else
        {
            RCLCPP_WARN_STREAM(this->get_logger(), "Unknown param \"" << param.get_name() << "\". Skipping.");
        }
    }

    return result;
}

void PlatoonDistanceEstimator::publish_leader_detection()
{
    // Lock both ego and target mutex
    std::unique_lock<std::mutex> ego_lock(this->ego_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> leader_lock(this->leader_mutex_, std::defer_lock);
    std::lock(ego_lock, leader_lock);

    PlatoonDetection::UniquePtr detection_msg = std::make_unique<PlatoonDetection>();
    detection_msg->header = this->ego_pose_->header;
    // Compute distance
    detection_msg->distance = this->leader_pose_->pose.position.x - this->ego_pose_->pose.position.x;
    // Add noise
    auto [noise, stddev] = this->compute_noise();
    detection_msg->distance += noise;
    // Add standard deviation
    detection_msg->stddev = stddev;
    // Publish the msg
    this->leader_detection_pub_->publish(std::move(detection_msg));
}

void PlatoonDistanceEstimator::publish_follower_detection()
{
    // Lock both ego and target mutex
    std::unique_lock<std::mutex> ego_lock(this->ego_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> follower_lock(this->follower_mutex_, std::defer_lock);
    std::lock(ego_lock, follower_lock);

    PlatoonDetection::UniquePtr detection_msg = std::make_unique<PlatoonDetection>();
    detection_msg->header = this->ego_pose_->header;
    // Compute distance
    detection_msg->distance = this->follower_pose_->pose.position.x - this->ego_pose_->pose.position.x;
    // Add noise
    auto [noise, stddev] = this->compute_noise();
    detection_msg->distance += noise;
    // Add standard deviation
    detection_msg->stddev = stddev;
    // Publish the msg
    this->follower_detection_pub_->publish(std::move(detection_msg));
}

std::tuple<double, double> PlatoonDistanceEstimator::compute_noise()
{
    double noise_stddev = this->distance_stddev_;
    // Check if the RNG gods want this measurement to fail
    if (this->uniform_distribution_(this->random_generator_) < this->bad_measurement_probability_)
    {
        // Add a random value to the measurement gaussian stddev
        noise_stddev += this->exponential_distribution_(this->random_generator_);
    }
    // Compute the gaussian noise value
    auto distance_gauss_distribution = std::normal_distribution<>(0.0, noise_stddev);
    double noise = distance_gauss_distribution(this->random_generator_);
    return std::tie(noise, noise_stddev);
}

void PlatoonDistanceEstimator::ego_pose_callback(const Pose::SharedPtr ego_pose_msg)
{
    // Lock mutex
    std::lock_guard<std::mutex> lock(this->ego_mutex_);
    this->ego_pose_ = ego_pose_msg;
}


void PlatoonDistanceEstimator::leader_pose_callback(const Pose::SharedPtr leader_pose_msg)
{
    // Lock mutex
    std::lock_guard<std::mutex> lock(this->leader_mutex_);
    this->leader_pose_ = leader_pose_msg;
}

void PlatoonDistanceEstimator::follower_pose_callback(const Pose::SharedPtr follower_pose_msg)
{
    // Lock mutex
    std::lock_guard<std::mutex> lock(this->follower_mutex_);
    this->follower_pose_ = follower_pose_msg;
}


}  // namespace fake_vehicle_detection

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(fake_vehicle_detection::PlatoonDistanceEstimator)
