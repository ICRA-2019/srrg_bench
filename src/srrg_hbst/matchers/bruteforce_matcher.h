#pragma once
#include "base_matcher.hpp"

namespace srrg_bench {

//! @class matcher implementing the OpenCV brute-force matching algorithm
class BruteforceMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  BruteforceMatcher(const uint32_t& minimum_distance_between_closure_images_,
                    const int32_t& norm_type_);

  //! @brief default destructor
  ~BruteforceMatcher();

  //! @brief prohibit default construction
  BruteforceMatcher() = delete;

//ds required interface
public:

  //! @brief database add function
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

  //! query function for results with descriptor associations
  void query(const cv::Mat& query_descriptors_,
             const ImageNumberQuery& image_number_,
             const uint32_t& maximum_descriptor_distance_,
             std::vector<ResultDescriptorMatching>& closures_);

//ds attributes
protected:

  //! @brief BF matcher
  cv::BFMatcher* _matcher;

  //! @brief bookkeeping: trained descriptors - for true, deep bruteforce matching
  std::map<ImageNumber, cv::Mat> _train_descriptors;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;
};
}
