#include "fake_vehicle_detection/platoon_multi_distance_estimator.hpp"

namespace fake_vehicle_detection
{

PlatoonMultiDistanceEstimator::PlatoonMultiDistanceEstimator() :
    rclcpp::Node("platoon_multi_distance_estimator"),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

PlatoonMultiDistanceEstimator::PlatoonMultiDistanceEstimator(const rclcpp::NodeOptions& options) :
    rclcpp::Node("platoon_multi_distance_estimator", options),
    // random_generator_(rd_())
    random_generator_(42)
{
    this->init();
}

PlatoonMultiDistanceEstimator::~PlatoonMultiDistanceEstimator()
{
    this->running_ = false;
    if (this->executor_thread_.joinable())
    {
        this->executor_thread_.join();
    }
}

void PlatoonMultiDistanceEstimator::init()
{
    using std::placeholders::_1;
    // Initialize and declare parameters
    this->distance_stddev_ = this->declare_parameter("distance_stddev", 0.05);
    this->bad_measurement_probability_ = this->declare_parameter("bad_measurement_probability", 0.05);

    // Descriptor for read_only parameters. These parameters cannot be changed (only overrided from yaml or launch args)
    rcl_interfaces::msg::ParameterDescriptor read_only_descriptor;
    read_only_descriptor.read_only = true;

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
                                                std::bind(&PlatoonMultiDistanceEstimator::set_param_callback, this, _1));

    // Initialize random distributions
    this->uniform_distribution_ = std::uniform_real_distribution<>(0.0, 1.0);
    this->exponential_distribution_ = std::exponential_distribution<>(2.5);

    // Publisher
    this->detections_pub_ =
        this->create_publisher<PlatoonDetectionArray>("~/detections", rclcpp::SensorDataQoS());

    // Create a reentrant callback_group for callbacks, so they can be called concurrently
    this->callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
    auto sub_options = rclcpp::SubscriptionOptions();
    sub_options.callback_group = this->callback_group_;

    // Subscribers
    this->ego_pose_sub_ = this->create_subscription<Pose>(
        "pose", rclcpp::SensorDataQoS(),
        std::bind(&PlatoonMultiDistanceEstimator::ego_pose_callback, this, _1), sub_options);

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
    this->executor_thread_ = std::thread(&PlatoonMultiDistanceEstimator::run, this);
}

void PlatoonMultiDistanceEstimator::run()
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
        PlatoonDetectionArray::UniquePtr detections = std::make_unique<PlatoonDetectionArray>();
        // Compute distance from ego to each received pose
        for (size_t i = 0; i < this->number_of_vehicles_; i++)
        {
            detections->header = this->ego_pose_->header;
            if (this->poses_received_[i])
            {
                detections->detections.push_back(compute_distance(i));
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

rcl_interfaces::msg::SetParametersResult PlatoonMultiDistanceEstimator::set_param_callback(const std::vector<rclcpp::Parameter>& params)
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

loco_framework::msg::PlatoonDetection PlatoonMultiDistanceEstimator::compute_distance(const size_t i)
{
    // Lock both ego and target mutex
    std::unique_lock<std::mutex> ego_lock(this->ego_mutex_, std::defer_lock);
    std::unique_lock<std::mutex> target_lock(this->pose_mutexes_[i], std::defer_lock);
    std::lock(ego_lock, target_lock);

    PlatoonDetection detection_msg;
    detection_msg.header = this->ego_pose_->header;
    // Compute distance
    detection_msg.distance = this->poses_[i].pose.position.x - this->ego_pose_->pose.position.x;
    // Add noise
    auto [noise, stddev] = this->compute_noise();
    detection_msg.distance += noise;
    // Add standard deviation
    detection_msg.stddev = stddev;
    return detection_msg;
}

std::tuple<double, double> PlatoonMultiDistanceEstimator::compute_noise()
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

void PlatoonMultiDistanceEstimator::ego_pose_callback(const Pose::SharedPtr ego_pose_msg)
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
RCLCPP_COMPONENTS_REGISTER_NODE(fake_vehicle_detection::PlatoonMultiDistanceEstimator)
