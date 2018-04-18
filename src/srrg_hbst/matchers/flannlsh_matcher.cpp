#include "flannlsh_matcher.h"

namespace srrg_bench {

FLANNLSHMatcher::FLANNLSHMatcher(const uint32_t& minimum_distance_between_closure_images_,
                                 const int32_t& table_number_,
                                 const int32_t& key_size_,
                                 const int32_t& multi_probe_level_): _matcher(new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(table_number_, key_size_, multi_probe_level_))),
                                                                     _minimum_distance_between_closure_images(minimum_distance_between_closure_images_) {
  _train_descriptor_details.clear();
  _image_numbers.clear();
  _durations_seconds_query_and_train.clear();
}

FLANNLSHMatcher::~FLANNLSHMatcher() {
  _train_descriptor_details.clear();
  _image_numbers.clear();
  delete _matcher;
}

void FLANNLSHMatcher::train(const cv::Mat& train_descriptors_,
                            const ImageNumberTrain& image_number_,
                            const std::vector<cv::KeyPoint>& train_keypoints_) {

  //ds add descriptors
  TIC(_time_begin);
  _matcher->add(std::vector<cv::Mat>(1, train_descriptors_));
  _matcher->train();
  _durations_seconds_query_and_train.back() += TOC(_time_begin).count();

  //ds bookkeep added descriptors
  _image_numbers.insert(std::make_pair(_train_descriptor_details.size(), image_number_));
  _train_descriptor_details.insert(std::make_pair(image_number_, std::make_pair(train_descriptors_.rows, std::set<int32_t>())));
}

void FLANNLSHMatcher::query(const cv::Mat& query_descriptors_,
                        const ImageNumberQuery& image_number_,
                        const uint32_t& maximum_distance_hamming_,
                        std::vector<ResultImageRetrieval>& closures_) {

  //ds match result handle: for each descriptor we get the k nearest neighbors
  std::vector<std::vector<cv::DMatch>> k_matches_per_descriptor;

  //ds match against database - this will trigger the flann matcher train method - integrating the trained descriptors
  TIC(_time_begin);
  _matcher->knnMatch(query_descriptors_, k_matches_per_descriptor, _train_descriptor_details.size());
  _durations_seconds_query_and_train.push_back(TOC(_time_begin).count());

  //ds clear bookkeeping to block N query descriptor to 1 train descriptor matches (manual crosscheck)
  for (std::pair<ImageNumberTrain, std::pair<uint64_t, std::set<int32_t>>> element: _train_descriptor_details) {
    element.second.second.clear();
  }

  //ds check for the best matching ratio over all neighbors
  std::multiset<uint32_t> number_of_matches_per_image;
  if (k_matches_per_descriptor.size() > 0) {
    for (const std::vector<cv::DMatch>& matches: k_matches_per_descriptor) {
      for (const cv::DMatch& match: matches) {

        //ds if the descriptor distance is acceptable and the match is available
        if (match.distance < maximum_distance_hamming_ && _train_descriptor_details[_image_numbers[match.imgIdx]].second.count(match.trainIdx) == 0) {

          //ds register and store it
          _train_descriptor_details[_image_numbers[match.imgIdx]].second.insert(match.trainIdx);
          number_of_matches_per_image.insert(_image_numbers[match.imgIdx]);
        }
      }
    }

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
  }
}
}
