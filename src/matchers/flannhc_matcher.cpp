#include "flannhc_matcher.h"

namespace srrg_bench {

FLANNHCMatcher::FLANNHCMatcher(const uint32_t& minimum_distance_between_closure_images_): _minimum_distance_between_closure_images(minimum_distance_between_closure_images_) {
  _durations_seconds_query_and_train.clear();
  _indices.clear();
  _added_descriptors.clear();
}

FLANNHCMatcher::~FLANNHCMatcher() {
  _indices.clear();
  for (flann::Matrix<DescriptorFLANN>& descriptors: _added_descriptors) {
    delete[] descriptors.ptr();
  }
  _added_descriptors.clear();
}

void FLANNHCMatcher::add(const cv::Mat& train_descriptors_,
                         const ImageNumberTrain& image_number_,
                         const std::vector<cv::KeyPoint>& train_keypoints_) {

}

void FLANNHCMatcher::train(const cv::Mat& train_descriptors_,
                           const ImageNumberTrain& image_number_,
                           const std::vector<cv::KeyPoint>& train_keypoints_) {

  //ds convert descriptors to flann format
  flann::Matrix<DescriptorFLANN> descriptors(new DescriptorFLANN[train_descriptors_.rows*DESCRIPTOR_SIZE_BYTES], train_descriptors_.rows, DESCRIPTOR_SIZE_BYTES);
  for (int64_t row = 0; row < train_descriptors_.rows; ++row) {
    for (uint64_t byte_index = 0; byte_index < DESCRIPTOR_SIZE_BYTES; ++byte_index) {
      descriptors[row][byte_index] = train_descriptors_.row(row).at<DescriptorFLANN>(byte_index);
    }
  }

  //ds add descriptors
  TIC(_time_begin);
  IndexFLANN index(descriptors, flann::HierarchicalClusteringIndexParams());
  index.buildIndex();
  _indices.insert(std::make_pair(image_number_, index));
  _durations_seconds_query_and_train.back() += TOC(_time_begin).count();
  _added_descriptors.push_back(descriptors);
}

void FLANNHCMatcher::query(const cv::Mat& query_descriptors_,
                           const ImageNumberQuery& image_number_,
                           const uint32_t& maximum_distance_hamming_,
                           std::vector<ResultImageRetrieval>& closures_) {

  //ds convert descriptors to flann format
  flann::Matrix<DescriptorFLANN> descriptors(new DescriptorFLANN[query_descriptors_.rows*DESCRIPTOR_SIZE_BYTES], query_descriptors_.rows, DESCRIPTOR_SIZE_BYTES);
  for (int64_t row = 0; row < query_descriptors_.rows; ++row) {
    for (uint64_t byte_index = 0; byte_index < DESCRIPTOR_SIZE_BYTES; ++byte_index) {
      descriptors[row][byte_index] = query_descriptors_.row(row).at<DescriptorFLANN>(byte_index);
    }
  }

  //ds match against database
  TIC(_time_begin);

  //ds FLANN result structures (assumed to be reset by flann)
  flann::Matrix<int32_t> indices(new int32_t[query_descriptors_.rows], descriptors.rows, 1);
  flann::Matrix<DistanceFLANN::ResultType> distances(new DistanceFLANN::ResultType[query_descriptors_.rows], descriptors.rows, 1);

  //ds matching result
  std::multiset<uint32_t> number_of_matches_per_image;

  //ds check every index
  for (const std::pair<ImageNumber, IndexFLANN>& index: _indices) {

    //ds query current index
    index.second.knnSearch(descriptors, indices, distances, 1, flann::SearchParams(-1));

    //ds check obtained matches
    for (uint64_t row = 0; row < distances.rows; ++row) {

      //ds if the descriptor distance is acceptable and the match is available (cross check in flann?)
      if (distances[row][0] < maximum_distance_hamming_) {

        //ds increase matching count for current image
        number_of_matches_per_image.insert(index.first);
      }
    }
  }
  delete[] indices.ptr();
  delete[] distances.ptr();
  _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());
  delete[] descriptors.ptr();

  //ds generate sortable score vector
  for (ImageNumberTrain image_number_train = 0; image_number_train < image_number_; ++image_number_train) {

    //ds if we can report the score for precision/recall evaluation
    if (image_number_ >= _minimum_distance_between_closure_images && image_number_train <= image_number_-_minimum_distance_between_closure_images) {

      //ds compute relative matching score
      const double score = static_cast<double>(number_of_matches_per_image.count(image_number_train))/query_descriptors_.rows;

      //ds add the closure
      closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_train)));
    }
  }

  //ds sort results in descending score
  std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
      {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
}
}
