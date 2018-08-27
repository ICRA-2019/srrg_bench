#pragma once
#include "base_matcher.hpp"

namespace srrg_bench {

//! @class matcher implementing the OpenCV FLANN LSH matching algorithm
class FLANNLSHMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  //! @param[in] table_number_
  //! @param[in] key_size_
  //! @param[in] multi_probe_level_
  FLANNLSHMatcher(const uint32_t& minimum_distance_between_closure_images_,
                  const int32_t& table_number_,
                  const int32_t& key_size_,
                  const int32_t& multi_probe_level_);

  //! @brief default destructor
  ~FLANNLSHMatcher();

  //! @brief prohibit default construction
  FLANNLSHMatcher() = delete;

//ds required interface
public:

  //! @brief database add function
  //! @param[in] train_descriptors_ collection of descriptors to be integrated into the database - no effect for HBST as we train directly while matching
  //! @param[in] image_number_ the image number associated with train_descriptors_
  virtual void add(const cv::Mat& train_descriptors_,
                   const ImageNumberTrain& image_number_,
                   const std::vector<cv::KeyPoint>& train_keypoints_);

  //! @brief default database index training function
  virtual void train();

  //! @brief database train function
  //! @param[in] train_descriptors_ collection of descriptors to be integrated into the database - no effect for HBST as we train directly while matching
  //! @param[in] image_number_ the image number associated with train_descriptors_
  virtual void train(const cv::Mat& train_descriptors_,
                     const ImageNumberTrain& image_number_,
                     const std::vector<cv::KeyPoint>& train_keypoints_);

  //! @brief database matching function
  //! @param[in] query_descriptors_ collection of descriptors to match against the train_descriptors_ collections of each past image
  //! @param[in] image_number_ the image number associated with query_descriptors_
  //! @param[in] maximum_distance_hamming_ maximum allowed hamming distance for a valid match
  //! @param[out] closures_ collection of image to image matches
  virtual void query(const cv::Mat& query_descriptors_,
                     const ImageNumberQuery& image_number_,
                     const uint32_t& maximum_distance_hamming_,
                     std::vector<ResultImageRetrieval>& closures_);

//ds attributes
protected:

  //! @brief FLANN matcher
  cv::FlannBasedMatcher* _matcher;

  //! @brief bookkeeping: mapping from matcher index to image_number (enabling query interspaces)
  std::map<int32_t, ImageNumber> _image_numbers;

  //! @brief bookkeeping: number of descriptors in images, match check list to block N query to 1 train descriptor matchings
  std::map<ImageNumberTrain, std::set<int32_t>> _train_descriptor_details;

  //! @brief bookkeeping: trained descriptors
  std::map<ImageNumber, cv::Mat> _train_descriptors;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;
};
}
