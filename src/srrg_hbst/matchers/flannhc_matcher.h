#pragma once
#include "base_matcher.hpp"
#include <flann/flann.hpp>
#include <flann/algorithms/hierarchical_clustering_index.h>

namespace srrg_bench {

typedef unsigned char DescriptorFLANN;
typedef flann::Hamming<DescriptorFLANN> DistanceFLANN;
typedef flann::HierarchicalClusteringIndex<DistanceFLANN> IndexFLANN;

//! @class matcher implementing the OpenCV FLANN LSH matching algorithm
class FLANNHCMatcher: public BaseMatcher {

//ds object life
public:

  //! @brief constructor
  //! @param[in] interspace_image_number_ image querying interspace
  //! @param[in] minimum_distance_between_closure_images_ minimum image number distance between closures
  //! @param[in] table_number_
  //! @param[in] key_size_
  //! @param[in] multi_probe_level_
  FLANNHCMatcher(const uint32_t& interspace_image_number_,
                 const uint32_t& minimum_distance_between_closure_images_);

  //! @brief default destructor
  ~FLANNHCMatcher();

  //! @brief prohibit default construction
  FLANNHCMatcher() = delete;

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

  //! @brief hierarchical clustered flann indices, one per stored image (TODO check for incremental implementation)
  std::map<ImageNumber, IndexFLANN> _indices;

  //! @brief added descriptor matrices (for deallocation)
  std::vector<flann::Matrix<DescriptorFLANN>> _added_descriptors;

  //! @brief query interspace
  uint32_t _interspace_image_number;

  //! @brief minimum image number distance between closures
  uint32_t _minimum_distance_between_closure_images;
};
}
