#include "hbst_matcher.h"

namespace srrg_bench {

HBSTMatcher::HBSTMatcher(const uint32_t& minimum_distance_between_closure_images_,
                         const srrg_hbst::SplittingStrategy& train_mode_): _minimum_distance_between_closure_images(minimum_distance_between_closure_images_),
                                                                           _train_mode(train_mode_) {
  clear();
}

HBSTMatcher::~HBSTMatcher() {
  clear();
}

void HBSTMatcher::train(const cv::Mat& train_descriptors_,
                        const ImageNumberTrain& image_number_,
                        const std::vector<cv::KeyPoint>& train_keypoints_) {
  //ds nothing to do -> matchAndAdd in query call
}

void HBSTMatcher::query(const cv::Mat& query_descriptors_,
                        const ImageNumberQuery& image_number_,
                        const uint32_t& maximum_distance_hamming_,
                        std::vector<ResultImageRetrieval>& closures_) {
  if (_database) {

    //ds match result handle
    Tree::MatchVectorMap matches;

    //ds obtain matchables (not timed being raw data, same for bow)
    const Tree::MatchableVector matchables(_database->getMatchablesWithIndex(query_descriptors_, image_number_));

    //ds match against database (tracking or match and add with simultaneous training)
    TIC(_time_begin);
    _database->matchAndAdd(matchables, matches, maximum_distance_hamming_, _train_mode);
    _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());

    //ds result evaluation
    for (const Tree::MatchVectorMapElement& match_vector: matches) {
      const ImageNumberTrain& image_number_reference = match_vector.first;

      //ds check if we can report the score for precision/recall evaluation
      if (image_number_ >= _minimum_distance_between_closure_images && image_number_reference <= image_number_-_minimum_distance_between_closure_images) {

        //ds compute score
        const double score = static_cast<double>(match_vector.second.size())/query_descriptors_.rows;

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_reference)));
      }
    }
  }
}

void HBSTMatcher::clear() {
  BaseMatcher::clear();
  _database = std::make_shared<Tree>();
}
}
