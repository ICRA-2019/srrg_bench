#include "loop_closure_evaluator.h"
#include <memory>

//ds custom descriptor types
#include "thirdparty/bold/bold.hpp"

namespace srrg_bench {

class CommandLineParameters {

  //ds object life
  public:

    CommandLineParameters() {}
    ~CommandLineParameters();

  //ds parsing
  public:

    //! @brief parses as many parameters as possible from command line arguments
    //! @param[in] argc_
    //! @param[in] argc_
    void parse(const int32_t& argc_, char** argv_);

    //! @brief validates currently parsed parameters
    //! @param[in] stream_
    //! @throws std::runtime_error
    void validate(std::ostream& stream_);

    //! @brief dumps full configuration to stream
    //! @param[in] stream_
    void write(std::ostream& stream_);

    //! @brief configures module - loads structures (e.g. loop closure evaluator) based on configuration
    //! @param[in] stream_
    //! @throws std::runtime_error
    void configure(std::ostream& stream_);

    //! descriptor computation
    void computeDescriptors(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& descriptors_);

    //! capped descriptor computation
    void computeDescriptors(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& descriptors_, const uint32_t& target_number_of_descriptors_);

    //! @brief constructs structures required for image region based descriptor augmentation
    void configurePositionAugmentation(const std::string& image_resolution_key_);

    //! visualization only
    void displayKeypoints(const cv::Mat& image_, const std::vector<cv::KeyPoint>& keypoints_) const;

  //ds TODO encapsulate
  public:

    //ds general configuration
    std::string method_name                     = "";
    std::string folder_images                   = "";
    std::string file_name_poses_ground_truth    = "";
    std::string file_name_closures_ground_truth = "";
    std::string parsing_mode                    = "kitti";
    uint32_t image_number_start                 = 0;
    uint32_t image_number_stop                  = 0;
    uint32_t number_of_images_to_process        = 0;
    uint32_t number_of_openmp_threads           = 4;

    //ds ground truth details
    uint32_t query_interspace                        = 1;
    uint32_t minimum_distance_between_closure_images = 500; //ds adjust this parameter accordingly if a closure gt is used
    double maximum_difference_position_meters        = 10;
    double maximum_difference_angle_radians          = M_PI/10;
    bool load_cross_datasets                         = false; //ds e.g. oxford datasets

    //ds method details
    std::string descriptor_type           = "brief"; //ds the descriptor type for bow has to be provided with DBOW2_DESCRIPTOR_TYPE in dbow_matcher.h
    int32_t distance_norm                 = cv::NORM_HAMMING;
    double maximum_descriptor_distance    = 25;
    uint32_t fast_detector_threshold      = 10;
    bool use_gui                          = false;
    uint32_t target_number_of_descriptors = 1000;
    double target_recall                  = 1;
    uint32_t training_delay_in_frames     = 0;

    //ds HBST specific
    uint32_t maximum_leaf_size  = 50;
    double maximum_partitioning = 0.1;
    bool use_random_splitting   = false;
    bool use_uneven_splitting   = false;
    uint32_t number_of_samples  = 1;
    uint32_t maximum_depth      = 5;

    //ds BoW specific
    std::string file_path_vocabulary = "";
    bool use_direct_index            = true;
    uint32_t direct_index_levels     = 2;
    bool compute_score_only          = false; //ds only check image scoring for precision computation (no associations computed)

    //ds LSH specific
    int32_t table_number      = 10;
    int32_t hash_key_size     = 20;
    int32_t multi_probe_level = 2;

    //ds lucia specific
    std::string file_name_image_timestamps = "";

    //ds oxford specific (multi sequence loading)
    std::string file_name_poses_ground_truth_cross = "";
    std::string folder_images_cross                = "";

    //ds nordland specific (multi video loading)
    cv::VideoCapture video_player_query;
    cv::VideoCapture video_player_reference;

    //ds custom descriptors
    std::shared_ptr<BOLD> bold_descriptor_handler;

    //! @brief augmentation properties
    uint32_t number_of_augmentation_bins_horizontal         = 0;
    uint32_t number_of_augmentation_bins_vertical           = 0;
    uint32_t number_of_augmented_bits                       = 0;
    uint32_t number_of_image_rows                           = 0;
    uint32_t number_of_image_cols                           = 0;
    uint32_t augmentation_weight                            = 0;

    //! @brief position augmentation mapping: [image_resolution_key]>[keypoint_row][keypoint_col]>[augmentation]
    //! @brief where image_resolution_key could be "240x320" (rows x cols)
    //! @brief a new image_resolution_key entry is generated for every new image encountered
    std::map<std::string, std::string**> mappings_image_coordinates_to_augmentation;

    //ds GUI
    double display_scale = 1.0;

  //ds instantiated objects
  public:

    //ds ground truth
    std::shared_ptr<LoopClosureEvaluator> evaluator;

    //ds feature handling
    cv::Ptr<cv::FeatureDetector> feature_detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
};
}
