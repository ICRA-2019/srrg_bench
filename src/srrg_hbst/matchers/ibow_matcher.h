#pragma once
#include "base_matcher.hpp"
#include "obindex2/binary_index.h"

namespace srrg_bench {

//! @class matcher implementing the OpenCV brute-force matching algorithm
class iBoWMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] interspace_image_number_ image querying interspace
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  iBoWMatcher(const uint32_t& interspace_image_number_,
              const uint32_t& minimum_distance_between_closure_images_,
              const unsigned k_ = 16,
              const unsigned s_ = 150,
              const unsigned t_ = 4,
              const obindex2::MergePolicy merge_policy_ = obindex2::MERGE_POLICY_AND,
              const bool purge_descriptors_ = true,
              const unsigned min_feat_apps_ = 3,
              const uint32_t& number_of_leaf_checks_ = 64);

  //! @brief default destructor
  ~iBoWMatcher();

  //! @brief prohibit default construction
  iBoWMatcher() = delete;

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

//ds attributes
protected:

  //! @brief image database
  obindex2::ImageIndex _index;

  //! @brief query interspace
  uint32_t _interspace_image_number;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;

  //! @brief total number of leaf checks for matching
  uint32_t _number_of_leaf_checks;
};
}
