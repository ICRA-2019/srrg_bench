#include "ibow_matcher.h"

namespace srrg_bench {

IBoWMatcher::IBoWMatcher(const uint32_t& minimum_distance_between_closure_images_,
                         const uint32_t& maximum_descriptor_distance_,
                         const unsigned k,
                         const unsigned s,
                         const unsigned t,
                         const obindex2::MergePolicy merge_policy,
                         const bool purge_descriptors,
                         const unsigned min_feat_apps,
                         const uint32_t& number_of_leaf_checks_): _maximum_descriptor_distance(maximum_descriptor_distance_),
                                                                  _index(obindex2::ImageIndex(k, s, t, merge_policy, purge_descriptors, min_feat_apps)),
                                                                  _minimum_distance_between_closure_images(minimum_distance_between_closure_images_),
                                                                  _number_of_leaf_checks(number_of_leaf_checks_) {
  _durations_seconds_query_and_train.clear();
}

  IBoWMatcher::~IBoWMatcher() {
}

void IBoWMatcher::add(const cv::Mat& train_descriptors_,
                      const ImageNumberTrain& image_number_,
                      const std::vector<cv::KeyPoint>& train_keypoints_) {
  TIC(_time_begin);

  //ds for all images except the first we need to match against the database before adding (incremental creation)
  if (_index.numImages() > 0) {

    //ds match against current database
    std::vector<std::vector<cv::DMatch>> matches_per_descriptor;
    _index.searchDescriptors(train_descriptors_, &matches_per_descriptor, 1, _number_of_leaf_checks);

    //ds refine matching result with 2-NN distance ratio check
    std::vector<cv::DMatch> matches;
    for (uint32_t u = 0; u < matches_per_descriptor.size(); u++) {
      if (matches_per_descriptor[u][0].distance < _maximum_descriptor_distance) {
        matches.push_back(matches_per_descriptor[u][0]);
      }
    }

    //ds integrate image after matching against database
    _index.addImage(image_number_, train_keypoints_, train_descriptors_, matches);
  } else {

    //ds add image without matching against database
    _index.addImage(image_number_, train_keypoints_, train_descriptors_);
  }
  _total_duration_add_seconds += TOC(_time_begin).count();
}

void IBoWMatcher::train() {
  TIC(_time_begin);
  _index.rebuild();
  _total_duration_train_seconds += TOC(_time_begin).count();
}

void IBoWMatcher::train(const cv::Mat& train_descriptors_,
                        const ImageNumberTrain& image_number_,
                        const std::vector<cv::KeyPoint>& train_keypoints_) {
  //ds TODO purge
}

void IBoWMatcher::query(const cv::Mat& query_descriptors_,
                        const ImageNumberQuery& image_number_,
                        const uint32_t& maximum_distance_hamming_,
                        std::vector<ResultImageRetrieval>& closures_) {
  closures_.clear();

  //ds match against all descriptor sets in database (assuming training happened in identical order)
  TIC(_time_begin);
  std::vector<std::vector<cv::DMatch>> matches_per_descriptor;
  _index.searchDescriptors(query_descriptors_, &matches_per_descriptor, 1, _number_of_leaf_checks);
  _total_duration_query_seconds += TOC(_time_begin).count();

  //ds matching result
  std::vector<cv::DMatch> matches_per_image;

  //ds for each image
  for (const std::vector<cv::DMatch>& matches: matches_per_descriptor) {
    if (!matches.empty()) {
      if (matches.front().distance < maximum_distance_hamming_) {
        matches_per_image.push_back(matches.front());
      }
    }
  }

  //ds if we obtained matches
  if (!matches_per_image.empty()) {

    //ds we look for similar images according to the good matches found
    std::vector<obindex2::ImageMatch> image_matches;
    TIC(_time_begin);
    _index.searchImages(query_descriptors_, matches_per_image, &image_matches);
    _total_duration_query_seconds += TOC(_time_begin).count();
    for (const obindex2::ImageMatch& image_match: image_matches) {
      const ImageNumberTrain image_number_train = image_match.image_id;

      //ds if we can report the score for precision/recall evaluation
      if ((image_number_ >= _minimum_distance_between_closure_images                  &&
          image_number_train <= image_number_-_minimum_distance_between_closure_images) ||
          _minimum_distance_between_closure_images == 0                                 ) {

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(image_match.score, ImageNumberAssociation(image_number_, image_number_train)));
      }
    }
  }

  //ds sort results in descending score
  std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
      {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
}
}
