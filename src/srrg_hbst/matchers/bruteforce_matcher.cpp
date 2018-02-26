#include "bruteforce_matcher.h"

namespace srrg_bench {

  BruteforceMatcher::BruteforceMatcher(const uint32_t& interspace_image_number_,
                                       const uint32_t& minimum_distance_between_closure_images_): _matcher(new cv::BFMatcher(cv::NORM_HAMMING, true)),
                                                                                                  _interspace_image_number(interspace_image_number_),
                                                                                                  _minimum_distance_between_closure_images(minimum_distance_between_closure_images_) {
    _durations_seconds_query_and_train.clear();
    _train_descriptors.clear();
  }

BruteforceMatcher::~BruteforceMatcher() {
  _train_descriptors.clear();
  delete _matcher;
}

void BruteforceMatcher::train(const cv::Mat& train_descriptors_,
                              const ImageNumberTrain& image_number_,
                              const std::vector<cv::KeyPoint>& train_keypoints_) {

  //ds add descriptors
  TIC(_time_begin);
  _train_descriptors.insert(std::make_pair(image_number_, train_descriptors_));
  _durations_seconds_query_and_train.back() += TOC(_time_begin).count();
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

    //ds compute individual matches
    std::vector<cv::DMatch> matches;
    _matcher->match(query_descriptors_, train_descriptors, matches);

    //ds threshold against maximum distance (cross checking is enabled so we get best-to-best descriptor associations)
    uint64_t number_of_matches = 0;
    for (const cv::DMatch& match: matches) {
      if (match.distance < maximum_distance_hamming_) {
        ++number_of_matches;
      }
    }

    //ds if the database entry is queriable for precision recall evaluation
    if (image_number_train%_interspace_image_number == 0) {

      //ds if we can report the score for precision/recall evaluation
      if (image_number_ >= _minimum_distance_between_closure_images && image_number_train <= image_number_-_minimum_distance_between_closure_images) {

        //ds compute relative matching score
        const double score = static_cast<double>(number_of_matches)/query_descriptors_.rows;

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_train)));
      }
    }
  }
  _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());
}

void BruteforceMatcher::query(const cv::Mat& query_descriptors_,
                              const ImageNumberQuery& image_number_,
                              const uint32_t& maximum_distance_hamming_,
                              std::vector<ResultDescriptorMatching>& closures_) {

  //ds match against all descriptor sets in database (assuming training happened in identical order)
  TIC(_time_begin);
  for (std::pair<ImageNumber, cv::Mat> train_descriptors_per_image: _train_descriptors) {
    const ImageNumber& image_number_train = train_descriptors_per_image.first;
    const cv::Mat& train_descriptors      = train_descriptors_per_image.second;

    //ds compute individual matches
    std::vector<cv::DMatch> matches;
    _matcher->match(query_descriptors_, train_descriptors, matches);

    //ds associations
    IndexAssociationVector descriptor_associations(0);

    //ds threshold against maximum distance (cross checking is enabled so we get best-to-best descriptor associations)
    for (const cv::DMatch& match: matches) {
      if (match.distance < maximum_distance_hamming_) {
        descriptor_associations.push_back(IndexAssociation(match.queryIdx, match.trainIdx));
      }
    }

    //ds if the database entry is queriable for precision recall evaluation
    if (image_number_train%_interspace_image_number == 0) {

      //ds if we can report the score for precision/recall evaluation
      if (image_number_ >= _minimum_distance_between_closure_images && image_number_train <= image_number_-_minimum_distance_between_closure_images) {

        //ds compute relative matching score
        const double score = static_cast<double>(descriptor_associations.size())/query_descriptors_.rows;

        //ds add the closure
        closures_.push_back(ResultDescriptorMatching(score, ImageNumberAssociation(image_number_, image_number_train), descriptor_associations));
      }
    }
  }
  _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());
}
}