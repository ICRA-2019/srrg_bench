#pragma once
#include "DBoW2/DBoW2.h"
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h"
#include "DVision/DVision.h"
#include "base_matcher.hpp"

using namespace DBoW2;

//ds bow descriptor type (only effective when chosen as method)
#define DBOW2_DESCRIPTOR_TYPE 1 //ds 0: BRIEF, 1: ORB, 2: BRISK

#if DBOW2_DESCRIPTOR_TYPE == 0
  #define DBOW2_DESCRIPTOR_CLASS FBrief
#elif DBOW2_DESCRIPTOR_TYPE == 1
  #define DBOW2_DESCRIPTOR_CLASS FORB
#elif DBOW2_DESCRIPTOR_TYPE == 2
  #define DBOW2_DESCRIPTOR_CLASS FBRISK
#endif

namespace srrg_bench {

//! @class matcher implementing the DBoW2 matching algorithm
class BoWMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  //! @param[in] file_path_vocabulary_
  //! @param[in] use_direct_index_
  //! @param[in] direct_index_levels_
  //! @param[in] compute_score_only_
  BoWMatcher(const uint32_t& minimum_distance_between_closure_images_,
             const std::string& file_path_vocabulary_,
             const bool& use_direct_index_ = false,
             const uint32_t& number_of_direct_index_levels_ = 2,
             const bool& compute_score_only_ = false);

  //! @brief default destructor
  ~BoWMatcher();

  //! @brief prohibit default construction
  BoWMatcher() = delete;

//ds required interface
public:

  //! @brief database add function
  //! @param[in] train_descriptors_ collection of descriptors to be integrated into the database
  //! @param[in] image_number_ the image number associated with train_descriptors_
  virtual void add(const cv::Mat& train_descriptors_,
                   const ImageNumberTrain& image_number_,
                   const std::vector<cv::KeyPoint>& train_keypoints_);

  //! @brief default database index training function
  virtual void train() {};

  //! @brief database train function
  //! @param[in] train_descriptors_ collection of descriptors to be integrated into the database
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

//ds helpers
protected:

  //! @brief transforms an opencv mat descriptor into a boost dynamic bitset used by dbow2
  //! @param[in] descriptor_cv_ the opencv input descriptor
  //! @param[out] descriptor_dbow2_ the dbow2 output descriptor
  void _setDescriptor(const cv::Mat& descriptor_cv_, FBrief::TDescriptor& descriptor_dbow2_) const;

  //! @brief snippet: https://github.com/dorian3d/DLoopDetector/blob/master/include/DLoopDetector/TemplatedLoopDetector.h
  //! @brief used for descriptor association computation
  template<class TDescriptor, class F>
  void _getMatches_neighratio(const std::vector<TDescriptor> &A, const std::vector<unsigned int> &i_A,
                              const std::vector<TDescriptor> &B, const std::vector<unsigned int> &i_B,
                              std::vector<unsigned int> &i_match_A, std::vector<unsigned int> &i_match_B,
                              const uint32_t& maximum_distance_hamming_) const;

//ds attributes
protected:

  //! @brief BoW vocabulary
  TemplatedVocabulary<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS> _vocabulary;

  //! @brief BoW database
  TemplatedDatabase<DBOW2_DESCRIPTOR_CLASS::TDescriptor, DBOW2_DESCRIPTOR_CLASS>* _database;

  //! @brief bookkeeping: mapping from matcher index to image_number
  std::map<int32_t, ImageNumber> _image_numbers;

  //! @brief direct index levels (required for association retrieval)
  uint32_t _number_of_direct_index_levels;

  //! @brief bookkeeping: raw descriptors (required for association retrieval)
  std::map<ImageNumber, std::vector<DBOW2_DESCRIPTOR_CLASS::TDescriptor>> _raw_descriptors_per_image;

  //! @brief bookkeeping: transformed BoW descriptors (required for querying and training)
  std::map<ImageNumber, BowVector> _bow_descriptors_per_image;

  //! @brief bookkeeping: BoW features (required for training and association retrieval)
  std::map<ImageNumber, FeatureVector> _bow_features_per_image;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;

  //! @brief compute only the image-to-image score for closure precision evaluation
  bool _compute_score_only;
};
}
