#pragma once
#include "srrg_hbst_types/binary_tree.hpp"
#include "base_matcher.hpp"

namespace srrg_bench {

typedef srrg_hbst::BinaryTree256 Tree; //no augmentation

//! @class matcher implementing the HBST matching algorithm
class HBSTMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] interspace_image_number_ image querying interspace
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  HBSTMatcher(const uint32_t& interspace_image_number_,
              const uint32_t& minimum_distance_between_closure_images_,
              const srrg_hbst::SplittingStrategy& train_mode_);

  //! @brief default destructor
  ~HBSTMatcher();

  //! @brief prohibit default construction
  HBSTMatcher() = delete;

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

  //! @brief resets the matcher and all structures - base method should be called from subclasses
  virtual void clear() override;

//ds attributes
protected:

  //! @brief HBST database
  std::shared_ptr<Tree> _database;

  //! @brief query interspace
  uint32_t _interspace_image_number;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;

  //! @brief leaf spawning strategy
  srrg_hbst::SplittingStrategy _train_mode;
};
}
