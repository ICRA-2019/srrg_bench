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

void HBSTMatcher::add(const cv::Mat& train_descriptors_,
                      const ImageNumberTrain& image_number_,
                      const std::vector<cv::KeyPoint>& train_keypoints_) {
  if (_database) {

    //ds obtain matchables (not timed being raw data)
    const Tree::MatchableVector matchables(_database->getMatchablesWithIndex(train_descriptors_, image_number_));

    TIC(_time_begin);
    _database->add(matchables);
    const double duration_seconds = TOC(_time_begin).count();
    _durations_seconds_query_and_train.push_back(duration_seconds);
    _total_duration_add_seconds += duration_seconds;
  }
}

void HBSTMatcher::train() {
  if (_database) {
    TIC(_time_begin);
    _database->train(srrg_hbst::SplitEven);
    const double duration_seconds = TOC(_time_begin).count();
    _durations_seconds_query_and_train.push_back(duration_seconds);
    _total_duration_train_seconds += duration_seconds;
  }
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

    //ds match against database
    TIC(_time_begin);
    _database->match(matchables, matches, maximum_distance_hamming_);
    const double duration_seconds = TOC(_time_begin).count();
    _durations_seconds_query_and_train.push_back(duration_seconds);
    _total_duration_query_seconds += duration_seconds;

    //ds result evaluation
    for (const Tree::MatchVectorMapElement& match_vector: matches) {
      const ImageNumberTrain& image_number_reference = match_vector.first;

      //ds if we can report the score for precision/recall evaluation
      if ((image_number_ >= _minimum_distance_between_closure_images                      &&
          image_number_reference <= image_number_-_minimum_distance_between_closure_images) ||
          _minimum_distance_between_closure_images == 0                                     ) {

        //ds compute score
        const double score = static_cast<double>(match_vector.second.size())/query_descriptors_.rows;

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_reference)));
      }
    }

    //ds sort results in descending score
    std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
        {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
  }
}

void HBSTMatcher::clear() {
  BaseMatcher::clear();
  _database = std::make_shared<Tree>();
}
}
