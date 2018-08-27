#include "bruteforce_matcher.h"

namespace srrg_bench {

BruteforceMatcher::BruteforceMatcher(const uint32_t& minimum_distance_between_closure_images_,
                                     const int32_t& norm_type_): _matcher(new cv::BFMatcher(norm_type_, true)),
                                                                 _minimum_distance_between_closure_images(minimum_distance_between_closure_images_) {
    _durations_seconds_query_and_train.clear();
    _train_descriptors.clear();
}

BruteforceMatcher::~BruteforceMatcher() {
  _train_descriptors.clear();
  delete _matcher;
}

void BruteforceMatcher::add(const cv::Mat& train_descriptors_,
                            const ImageNumberTrain& image_number_,
                            const std::vector<cv::KeyPoint>& train_keypoints_) {
  TIC(_time_begin);
  _train_descriptors.insert(std::make_pair(image_number_, train_descriptors_));
  const double duration_seconds = TOC(_time_begin).count();
  _durations_seconds_query_and_train.push_back(duration_seconds);
  _total_duration_add_seconds += duration_seconds;
}

void BruteforceMatcher::train(const cv::Mat& train_descriptors_,
                              const ImageNumberTrain& image_number_,
                              const std::vector<cv::KeyPoint>& train_keypoints_) {
  TIC(_time_begin);
  _train_descriptors.insert(std::make_pair(image_number_, train_descriptors_));
  const double duration_seconds = TOC(_time_begin).count();
  _durations_seconds_query_and_train.push_back(duration_seconds);
  _total_duration_train_seconds += duration_seconds;
}

void BruteforceMatcher::query(const cv::Mat& query_descriptors_,
                              const ImageNumberQuery& image_number_,
                              const uint32_t& maximum_distance_hamming_,
                              std::vector<ResultImageRetrieval>& closures_) {

  //ds match against all descriptor sets in database (assuming training happened in identical order)
  TIC(_time_begin);
  for (std::pair<ImageNumber, cv::Mat> train_descriptors_per_image: _train_descriptors) {
    const ImageNumber& image_number_train = train_descriptors_per_image.first;
    const cv::Mat& train_descriptors      = train_descriptors_per_image.second;
    uint64_t number_of_matches            = 0;

    //ds if matching is possible
    if (train_descriptors.rows > 0 && query_descriptors_.rows > 0) {

      //ds compute individual matches
      std::vector<cv::DMatch> matches;
      _matcher->match(query_descriptors_, train_descriptors, matches);

      //ds threshold against maximum distance (cross checking is enabled so we get best-to-best descriptor associations)
      for (const cv::DMatch& match: matches) {
        if (match.distance < maximum_distance_hamming_) {
          ++number_of_matches;
        }
      }
    }

    //ds if we can report the score for precision/recall evaluation
    if ((image_number_ >= _minimum_distance_between_closure_images                  &&
        image_number_train <= image_number_-_minimum_distance_between_closure_images) ||
        _minimum_distance_between_closure_images == 0                                 ) {

      //ds compute relative matching score
      const double score = static_cast<double>(number_of_matches)/query_descriptors_.rows;

      //ds add the closure
      closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_train)));
    }
  }
  const double duration_seconds = TOC(_time_begin).count();
  _durations_seconds_query_and_train.push_back(duration_seconds);
  _total_duration_query_seconds += duration_seconds;

  //ds sort results in descending score
  std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
      {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
}

void BruteforceMatcher::query(const cv::Mat& query_descriptors_,
                              const ImageNumberQuery& image_number_,
                              const uint32_t& maximum_descriptor_distance_,
                              std::vector<ResultDescriptorMatching>& closures_) {

  //ds match against all descriptor sets in database (assuming training happened in identical order)
  TIC(_time_begin);
  for (std::pair<ImageNumber, cv::Mat> train_descriptors_per_image: _train_descriptors) {
    const ImageNumber& image_number_train = train_descriptors_per_image.first;
    const cv::Mat& train_descriptors      = train_descriptors_per_image.second;

    //ds if we can report the score for precision/recall evaluation
    if (image_number_ >= _minimum_distance_between_closure_images && image_number_train <= image_number_-_minimum_distance_between_closure_images) {

      //ds compute individual matches
      std::vector<cv::DMatch> matches;
      _matcher->match(query_descriptors_, train_descriptors, matches);

      //ds associations
      IndexAssociationVector descriptor_associations(0);

      //ds threshold against maximum distance (cross checking is enabled so we get best-to-best descriptor associations)
      for (const cv::DMatch& match: matches) {
        if (match.distance < maximum_descriptor_distance_) {
          descriptor_associations.push_back(IndexAssociation(match.queryIdx, match.trainIdx));
        }
      }

      //ds compute relative matching score
      const double score = static_cast<double>(descriptor_associations.size())/query_descriptors_.rows;

      //ds add the closure
      closures_.push_back(ResultDescriptorMatching(score, ImageNumberAssociation(image_number_, image_number_train), descriptor_associations));
    }
  }
  const double duration_seconds = TOC(_time_begin).count();
  _durations_seconds_query_and_train.push_back(duration_seconds);
  _total_duration_query_seconds += duration_seconds;
}
}
