#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <memory>
#include <Eigen/Geometry>
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

//ds easy logging macro
#define LOG_VARIABLE(VARIABLE_) \
  std::cerr << #VARIABLE_ << ": '" << VARIABLE_ << "'" << std::endl;

//ds easy logging macro - living from your expressiveness
#define WRITE_VARIABLE(STREAM_, VARIABLE_) \
  STREAM_ << #VARIABLE_ << ": '" << VARIABLE_ << "'" << std::endl

#define BAR "---------------------------------------------------------------------------------"

//ds descriptor configuration - default
#ifndef SRRG_BENCH_DESCRIPTOR_SIZE_BYTES
  #define SRRG_BENCH_DESCRIPTOR_SIZE_BYTES 32
#endif

//ds configure size
#define DESCRIPTOR_SIZE_BYTES SRRG_BENCH_DESCRIPTOR_SIZE_BYTES
#define DESCRIPTOR_SIZE_BITS DESCRIPTOR_SIZE_BYTES*8



namespace srrg_bench {

typedef uint32_t ImageNumber;
typedef uint32_t ImageNumberQuery;
typedef uint32_t ImageNumberTrain;
struct ImageNumberAssociation {
  ImageNumberAssociation(): query(0), train(0), valid(false) {}
  ImageNumberAssociation(const ImageNumberQuery& query_, const ImageNumberTrain& reference_): query(query_), train(reference_), valid(false) {}
  ImageNumberQuery query;
  ImageNumberTrain train;
  bool valid;
};
typedef std::vector<ImageNumberAssociation> ImagePairVector;

struct IndexAssociation {
  IndexAssociation(): index_query(0),
                      index_train(0) {}
  IndexAssociation(const uint64_t& index_query_,
                   const uint64_t& index_train_): index_query(index_query_),
                                                  index_train(index_train_) {}
  uint64_t index_query;
  uint64_t index_train;
};
typedef std::vector<IndexAssociation> IndexAssociationVector;

//! @brief search result for only image retrieval
struct ResultImageRetrieval {
    ResultImageRetrieval(const double& number_of_matches_relative_,
                         const ImageNumberAssociation& image_association_): number_of_matches_relative(number_of_matches_relative_),
                                                                            image_association(image_association_) {}

    ResultImageRetrieval(): number_of_matches_relative(0),
                            image_association(ImageNumberAssociation()) {}

    double number_of_matches_relative;
    ImageNumberAssociation image_association;
};

//! @brief descriptor matching
struct ResultDescriptorMatching {
    ResultDescriptorMatching(const double& number_of_matches_relative_,
                             const ImageNumberAssociation& image_association_,
                             const IndexAssociationVector& descriptor_associations_): result_image_retrieval(ResultImageRetrieval(number_of_matches_relative_, image_association_)),
                                                                                      descriptor_associations(descriptor_associations_),
                                                                                      number_of_matches_relative_verified(0) {}

    ResultDescriptorMatching(): result_image_retrieval(ResultImageRetrieval()),
                                descriptor_associations(IndexAssociationVector()),
                                number_of_matches_relative_verified(0) {}

    //ds image retrieval result
    ResultImageRetrieval result_image_retrieval;

    //ds optional, descriptor associations (indexes)
    IndexAssociationVector descriptor_associations;
    double number_of_matches_relative_verified;
};

struct PoseWithTimestamp {
public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseWithTimestamp(const Eigen::Isometry3d& pose_, const double& timestamp_seconds_): pose(pose_), timestamp_seconds(timestamp_seconds_) {}
  Eigen::Isometry3d pose;
  double timestamp_seconds;
};
typedef std::vector<PoseWithTimestamp, Eigen::aligned_allocator<PoseWithTimestamp>> PoseWithTimestampVector;

struct ImageFileWithTimestamp {
  ImageFileWithTimestamp(const std::string& file_name_, const double& timestamp_seconds_): file_name(file_name_), timestamp_seconds(timestamp_seconds_) {}
  std::string file_name;
  double timestamp_seconds;
};
typedef std::vector<ImageFileWithTimestamp> ImageFileWithTimestampVector;

//! @struct image/pose construct
struct ImageWithPose {
public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImageWithPose(const std::string& file_path_, const ImageNumber& image_number_, const Eigen::Isometry3d& pose_): file_path(file_path_),
                                                                                                                  image_number(image_number_),
                                                                                                                  pose(pose_) {}
  std::string file_path;
  ImageNumber image_number;
  Eigen::Isometry3d pose;

  //ds optional
  std::string file_name;
  std::string file_type;
  std::string file_path_origin;
};

typedef std::map<ImageNumberQuery, std::multiset<ImageNumberTrain>> ClosureMap;
typedef std::pair<ImageNumberQuery, std::multiset<ImageNumberTrain>> ClosureMapElement;

class LoopClosureEvaluator {
public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  //! @brief default constructor (empty)
  LoopClosureEvaluator();

  //! @brief default destructor (frees image with pose instances, interlinked in memory for closures)
  ~LoopClosureEvaluator();

public:

  void loadImagesWithPosesFromFileKITTI(const std::string& file_name_poses_ground_truth_, const std::string& folder_images_);

  void loadImagesWithPosesFromFileMalaga(const std::string& file_name_poses_ground_truth_, const std::string& images_folder_);

  void loadImagesWithPosesFromFileLucia(const std::string& file_name_poses_ground_truth_, const std::string& file_name_images_with_timestamps_);

  //! @brief oxford dataset loading: requiring at least two separate dataset sequences (season closing possible)
  void loadImagesWithPosesFromFileOxford(const std::string& file_name_poses_ground_truth_0_,
                                         const std::string& images_folder_0_,
                                         const std::string& file_name_poses_ground_truth_1_ = "",
                                         const std::string& images_folder_1_ = "");

  //! @brief nordland dataset loading: requiring at least two separate dataset sequences (season closing possible)
  void loadImagesWithPosesFromFileNordland(const std::string& file_name_poses_ground_truth_query_,
                                           const std::string& images_folder_query_,
                                           const std::string& file_name_poses_ground_reference_,
                                           const std::string& images_folder_reference_);

  void loadImagesFromDirectoryZubud(const std::string& directory_query_, const std::string& directory_reference_);

  void loadImagesFromDirectoryHolidays(const std::string& directory_images_,
                                       const std::string& file_name_ground_truth_mapping_);

  void loadImagesFromDirectoryOxford(const std::string& directory_query_, const std::string& directory_reference_, const std::string& parsing_mode_ = "");

  void computeLoopClosureFeasibilityMap(const uint32_t& image_number_start_                      = 0,
                                        const uint32_t& image_number_stop_                       = 0,
                                        const uint32_t& interspace_image_number_                 = 1,
                                        const double& maximum_difference_position_meters_        = 25,
                                        const double& maximum_difference_angle_radians_          = M_PI/10,
                                        const uint32_t& minimum_distance_between_closure_images_ = 500);

  void computeLoopClosureFeasibilityMap(const std::string& file_name_ground_truth_mapping_, const char& separator_);

  std::pair<double, double> getPrecisionRecall(ImagePairVector& reported_closures_, const double& target_recall_);

  std::vector<std::pair<double, double>> computePrecisionRecallCurve(std::vector<ResultImageRetrieval>& reported_closures_,
                                                                     double& maximum_f1_score_,
                                                                     const double& target_recall_ = 1.0,
                                                                     const std::string& file_name_ = "");

  //ds this method has been adapted from: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp
  double computeAveragePrecision(const std::vector<ResultImageRetrieval>& ranked_reference_image_list_,
                                 const std::multiset<ImageNumberTrain>& valid_reference_image_list_) const;

  void loadClosures(const std::string& file_name_closure_map_,
                    const uint32_t& image_number_start_                     = 0,
                    const uint32_t& image_number_stop_                      = 0,
                    const uint32_t& interspace_image_number_                = 1,
                    const double& maximum_difference_position_meters_       = 25,
                    const double& maximum_difference_angle_radians_         = M_PI/10,
                    const int32_t& minimum_distance_between_closure_images_ = 500);

//ds getters/setters
public:

  const uint32_t& numberOfImageRows() const {return _number_of_image_rows;}
  const uint32_t& numberOfImageCols() const {return _number_of_image_cols;}
  const ImageNumberQuery numberOfImages() const {return _image_poses_ground_truth.size();}
  const std::vector<ImageWithPose*>& imagePosesGroundTruth() const {return _image_poses_ground_truth;}
  const std::vector<ImageWithPose*>& imagePosesQuery() const {return _image_poses_query;}
  const ClosureMap& closureFeasibilityMap() const {return _closure_feasability_map;}
  const ImageNumber& totalNumberOfValidClosures() const {return _total_number_of_valid_closures;}
  const std::set<const ImageWithPose*>& validQueryImagesWithPoses() const {return _valid_query_image_numbers;}
  const std::set<const ImageWithPose*>& validTrainImagesWithPoses() const {return _valid_train_image_numbers;}
  const std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>>& validClosures() const {return _valid_closures;}
  const std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>>& invalidClosures() const {return _invalid_closures;}
  const double& trajectoryLengthMeters() const {return _trajectory_length_meters;}

//ds helpers
protected:

  PoseWithTimestampVector _getPosesFromGPSOxford(const std::string& file_name_poses_ground_truth_) const;
  ImageFileWithTimestampVector _getImageFilesFromGPSOxford(const std::string& folder_images_) const;
  void _initializeImageConfiguration(const std::string& image_file_name_, const bool& bayer_decoding_ = false);
  PoseWithTimestampVector _getPosesFromGPSNordland(const std::string& file_name_poses_) const;
  void _loadImagePathsFromDirectory(const std::string& directory_,
                                    const std::string& image_file_name_,
                                    std::vector<std::string>& image_paths_) const;

private:

  //! @brief image dimensions
  uint32_t _number_of_image_rows = 0;
  uint32_t _number_of_image_cols = 0;

  //! @brief different image file name probing modes (e.g. KITTI SRRG versus KITTI raw)
  uint32_t _image_file_name_mode = 0;

  //! @brief images linked to poses (potentially interpolated)
  std::vector<ImageWithPose*> _image_poses_ground_truth;

  //! @brief separate query images
  std::vector<ImageWithPose*> _image_poses_query;

  //! @brief parsing mode (dataset)
  std::string parsing_mode = "";

  //! @brief closure map: valid query to train
  ClosureMap _closure_feasability_map;

  //! @brief closure map: valid query to train (Brute-force matching filtered)
  ClosureMap _closure_map_bf;

  //! @brief total number of closures (query - train pairs)
  ImageNumber _total_number_of_valid_closures = 0;

  //! @brief visualization only: feasible query image numbers
  std::set<const ImageWithPose*> _valid_query_image_numbers;

  //! @brief visualization only: feasible train image numbers
  std::set<const ImageWithPose*> _valid_train_image_numbers;

  //! @brief visualization only: valid reported closures
  std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> _valid_closures;

  //! @brief visualization only: invalid reported closures
  std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> _invalid_closures;

  //! @brief visualization only: checked if recall for closure display is reached
  bool _reached_target_display_recall = false;

  //! @brief info only: total trajectory length
  double _trajectory_length_meters = 0;
};
}
