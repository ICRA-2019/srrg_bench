#include "ibow_matcher.h"

namespace srrg_bench {

  iBoWMatcher::iBoWMatcher(const uint32_t& minimum_distance_between_closure_images_,
                           const unsigned k,
                           const unsigned s,
                           const unsigned t,
                           const obindex2::MergePolicy merge_policy,
                           const bool purge_descriptors,
                           const unsigned min_feat_apps,
                           const uint32_t& number_of_leaf_checks_): _index(obindex2::ImageIndex(k, s, t, merge_policy, purge_descriptors, min_feat_apps)),
                                                                    _minimum_distance_between_closure_images(minimum_distance_between_closure_images_),
                                                                    _number_of_leaf_checks(number_of_leaf_checks_) {
    _durations_seconds_query_and_train.clear();
  }

  iBoWMatcher::~iBoWMatcher() {
}

void iBoWMatcher::add(const cv::Mat& train_descriptors_,
                        const ImageNumberTrain& image_number_,
                        const std::vector<cv::KeyPoint>& train_keypoints_) {

}

void iBoWMatcher::train(const cv::Mat& train_descriptors_,
                        const ImageNumberTrain& image_number_,
                        const std::vector<cv::KeyPoint>& train_keypoints_) {

  //ds add descriptors - ibow accepts only continuous image numbering!
  TIC(_time_begin);
  _index.addImage(image_number_, train_keypoints_, train_descriptors_);
  _durations_seconds_query_and_train.back() += TOC(_time_begin).count();
}

void iBoWMatcher::query(const cv::Mat& query_descriptors_,
                        const ImageNumberQuery& image_number_,
                        const uint32_t& maximum_distance_hamming_,
                        std::vector<ResultImageRetrieval>& closures_) {
  closures_.clear();

  //ds match against all descriptor sets in database (assuming training happened in identical order)
  TIC(_time_begin);
  std::vector<std::vector<cv::DMatch>> matches_per_descriptor;
  _index.searchDescriptors(query_descriptors_, &matches_per_descriptor, _index.numImages(), _number_of_leaf_checks);
  _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());

  //ds matching result
  std::vector<cv::DMatch> matches_per_image;

  //ds for each image
  for (const std::vector<cv::DMatch>& matches: matches_per_descriptor) {
    if (!matches.empty()) {

      //ds compute number of valid matches (votes) per image
      for (const cv::DMatch& match: matches) {
        if (match.distance < maximum_distance_hamming_) {
          matches_per_image.push_back(match);
        }
      }
    }
  }

  //ds if we obtained matches
  if (!matches_per_image.empty()) {

    //ds we look for similar images according to the good matches found
    std::vector<obindex2::ImageMatch> image_matches;
    TIC(_time_begin);
    _index.searchImages(query_descriptors_, matches_per_image, &image_matches);
    _durations_seconds_query_and_train.back() += TOC(_time_begin).count();
    for (const obindex2::ImageMatch& image_match: image_matches) {
      const ImageNumberTrain image_number_train = image_match.image_id;

      //ds if we can report the score for precision/recall evaluation
      if (image_number_ >= _minimum_distance_between_closure_images && image_number_train <= image_number_-_minimum_distance_between_closure_images) {

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(image_match.score, ImageNumberAssociation(image_number_, image_number_train)));
      }
    }
  }
}
}
