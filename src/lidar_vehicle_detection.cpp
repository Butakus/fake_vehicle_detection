#include "fake_vehicle_detection/lidar_vehicle_detection.hpp"

namespace fake_vehicle_detection
{

LidarVehicleDetection::LidarVehicleDetection() :
    rclcpp::Node("lidar_vehicle_detection"),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

LidarVehicleDetection::LidarVehicleDetection(const rclcpp::NodeOptions& options) :
    rclcpp::Node("lidar_vehicle_detection", options),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

LidarVehicleDetection::~LidarVehicleDetection()
{
    this->running_ = false;
    if (this->executor_thread_.joinable())
    {
        this->executor_thread_.join();
    }
}

void LidarVehicleDetection::init()
{
    using std::placeholders::_1;
    // Initialize and declare parameters
    this->bad_measurement_probability_ = this->declare_parameter("bad_measurement_probability", 0.05);

    // Descriptor for read_only parameters. These parameters cannot be changed (only overrided from yaml or launch args)
    rcl_interfaces::msg::ParameterDescriptor read_only_descriptor;
    read_only_descriptor.read_only = true;

    this->max_detection_range_ = this->declare_parameter("max_detection_range", 50.0, read_only_descriptor);
    this->rate_ = this->declare_parameter("rate", 20.0, read_only_descriptor);
    std::vector<std::string> pose_topics;
    pose_topics = this->declare_parameter("pose_topics", pose_topics, read_only_descriptor);

    // Check if topic lists are set
    if (pose_topics.size() == 0)
    {
        RCLCPP_ERROR(this->get_logger(), "Topic parameters not set");
        rclcpp::shutdown();
        return;
    }
    this->number_of_vehicles_ = pose_topics.size();

    // Set callback to handle parameter setting
    this->set_param_callback_handler_ = this->add_on_set_parameters_callback(
                                                std::bind(&LidarVehicleDetection::set_param_callback, this, _1));

    // Initialize random distributions
    this->uniform_distribution_ = std::uniform_real_distribution<>(0.0, 1.0);
    this->exponential_distribution_ = std::exponential_distribution<>(2.0);

    // Publisher
    this->detections_pub_ =
        this->create_publisher<DetectionArray>("~/detections", rclcpp::SensorDataQoS());

    // Create a reentrant callback_group for callbacks, so they can be called concurrently
    this->callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    auto sub_options = rclcpp::SubscriptionOptions();
    sub_options.callback_group = this->callback_group_;

    // Subscribers
    this->ego_pose_sub_ = this->create_subscription<Pose>(
        "pose", rclcpp::SensorDataQoS(),
        std::bind(&LidarVehicleDetection::ego_pose_callback, this, _1), sub_options);

    // Initialize state vectors
    this->poses_.resize(this->number_of_vehicles_);
    this->poses_received_.resize(this->number_of_vehicles_, false);

    // Initialize mutexes
    std::vector<std::mutex> temp_pose_mutexes(this->number_of_vehicles_);
    this->pose_mutexes_.swap(temp_pose_mutexes);

    // Define pose callbacks and initialize subscribers
    // Subscribers
    for (std::size_t i = 0; i < this->number_of_vehicles_; ++i)
    {
        // Create pose callback lambda with agent index
        auto pose_callback = [i, this](const Pose::SharedPtr msg) -> void
        {
            // Lock mutex
            std::lock_guard<std::mutex>(this->pose_mutexes_[i]);
            // Store the odom in the vector by the vehicle index
            this->poses_[i] = *msg;
            this->poses_received_[i] = true;
        };
        // Create and store odom subscriber
        rclcpp::Subscription<Pose>::SharedPtr pose_sub =
            this->create_subscription<Pose>(pose_topics[i], rclcpp::SensorDataQoS(),
                                            pose_callback, sub_options);
        this->pose_subs_.push_back(pose_sub);
    }

    // Start the main execution thread to update and publish the state
    this->executor_thread_ = std::thread(&LidarVehicleDetection::run, this);
}

void LidarVehicleDetection::run()
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
        DetectionArray::UniquePtr detections = std::make_unique<DetectionArray>();
        // Compute distance from ego to each received pose
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            detections->header = this->ego_pose_->header;
            if (this->poses_received_[i])
            {
                if (compute_distance(this->ego_pose_->pose, this->poses_[i].pose) <= this->max_detection_range_)
                detections->detections.push_back(compute_detection(i));
            }
        }
        // Publish distances
        if (detections->detections.size() > 0)
        {
            this->detections_pub_->publish(std::move(detections));
        }
        rate.sleep();
    }
}

rcl_interfaces::msg::SetParametersResult LidarVehicleDetection::set_param_callback(const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & param : params)
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Set param " << param.get_name() << ": " << param.value_to_string());
        if (param.get_name() == "bad_measurement_probability")
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

geometry_msgs::msg::PoseWithCovarianceStamped LidarVehicleDetection::compute_detection(const size_t i)
{
    // Lock both ego and target mutex
    std::unique_lock<std::mutex> ego_lock(this->ego_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> target_lock(this->pose_mutexes_[i], std::defer_lock);
    std::lock(ego_lock, target_lock);

    Detection detection_msg;
    detection_msg.header = this->ego_pose_->header;
    // Compute detection (relative transformation between poses)
    detection_msg.pose.pose = relative_transform(this->ego_pose_->pose, this->poses_[i].pose);
    // Compute distance and yaw angle
    double distance = compute_distance(this->ego_pose_->pose, this->poses_[i].pose);
    double angle = yaw_from_quaternion<double>(detection_msg.pose.pose.orientation);
    // Compute noise covariance
    Eigen::Matrix<double, 2, 2> noise_cov = this->get_noise_covariance(distance, angle);
    // Generate noise
    Eigen::Vector2d noise = this->generate_noise(noise_cov);
    // Add noise to detection
    detection_msg.pose.pose.position.x += noise(0);
    detection_msg.pose.pose.position.y += noise(1);
    // Add covariance
    std::array<std::array<double, 6>, 6> noise_cov_6{};
    noise_cov_6[0][0] = noise_cov(0, 0);
    noise_cov_6[0][1] = noise_cov(0, 1);
    noise_cov_6[1][0] = noise_cov(1, 0);
    noise_cov_6[1][1] = noise_cov(1, 1);
    detection_msg.pose.covariance = tf2::covarianceNestedToRowMajor(noise_cov_6);

    return detection_msg;
}

Eigen::Matrix<double, 2, 2> LidarVehicleDetection::get_noise_covariance(const double distance, const double angle)
{
    // Measurement model adapted from LPO landmark detection model
    double distance_std = 0.00055 * distance + 0.0008;
    double angle_std = 0.001; // rad
    double angle_std_scaled = distance * std::tan(angle_std);

    // Check if the RNG gods want this measurement to fail
    if (this->uniform_distribution_(this->random_generator_) < this->bad_measurement_probability_)
    {
        // Add a random value to the distance stddev
        distance_std += this->exponential_distribution_(this->random_generator_);
    }


    // Convert covariance from polar coordinates to cartesian
    Eigen::Matrix<double, 2, 2> polar_cov;
    polar_cov << distance_std*distance_std, 0.0,
                 0.0, angle_std_scaled*angle_std_scaled;

    Eigen::Matrix<double, 2, 2> R;
    R << std::cos(angle), -std::sin(angle),
         std::sin(angle), std::cos(angle);

    Eigen::Matrix<double, 2, 2> cartesian_cov = R * polar_cov * R.inverse();
    return cartesian_cov;
}

Eigen::Vector2d LidarVehicleDetection::generate_noise(const Eigen::Matrix<double, 2, 2>& noise_covariance)
{
    // Generate noise sample using multivariate normal distribution
    Eigen::Vector2d noise_mean {0.0, 0.0};
    MultivariateNormal<double, 2> mvn(noise_mean, noise_covariance);

    return mvn.sample(this->random_generator_);
}

void LidarVehicleDetection::ego_pose_callback(const Pose::SharedPtr ego_pose_msg)
{
    // Lock mutex
    std::lock_guard<std::mutex> lock(this->ego_mutex_);
    this->ego_pose_ = ego_pose_msg;
}


}  // namespace fake_vehicle_detection

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(fake_vehicle_detection::LidarVehicleDetection)
