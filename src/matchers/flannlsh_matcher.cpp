#include "flannlsh_matcher.h"

namespace srrg_bench {

FLANNLSHMatcher::FLANNLSHMatcher(const uint32_t& minimum_distance_between_closure_images_,
                                 const int32_t& table_number_,
                                 const int32_t& key_size_,
                                 const int32_t& multi_probe_level_): _matcher(new cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(table_number_, key_size_, multi_probe_level_),
                                                                                                        cv::makePtr<cv::flann::SearchParams>(32))),
                                                                     _minimum_distance_between_closure_images(minimum_distance_between_closure_images_) {
  _train_descriptor_details.clear();
  _image_numbers.clear();
}

FLANNLSHMatcher::~FLANNLSHMatcher() {
  _train_descriptor_details.clear();
  _image_numbers.clear();
  delete _matcher;
}

void FLANNLSHMatcher::add(const cv::Mat& train_descriptors_,
                          const ImageNumberTrain& image_number_,
                          const std::vector<cv::KeyPoint>& train_keypoints_) {

  //ds add descriptors
  TIC(_time_begin);
  _matcher->add(std::vector<cv::Mat>(1, train_descriptors_));

  //ds bookkeep added descriptors
  _image_numbers.insert(std::make_pair(_train_descriptor_details.size(), image_number_));
  _train_descriptor_details.insert(std::make_pair(image_number_, std::set<int32_t>()));
  _total_duration_add_seconds += TOC(_time_begin).count();
}

void FLANNLSHMatcher::train() {
  TIC(_time_begin);
  _matcher->train();
  _total_duration_train_seconds += TOC(_time_begin).count();
}

void FLANNLSHMatcher::train(const cv::Mat& train_descriptors_,
                            const ImageNumberTrain& image_number_,
                            const std::vector<cv::KeyPoint>& train_keypoints_) {
  //ds TODO purge

  //ds add descriptors
  TIC(_time_begin);
  _matcher->add(std::vector<cv::Mat>(1, train_descriptors_));
  _matcher->train();
  _durations_seconds_query_and_train.back() += TOC(_time_begin).count();

  //ds bookkeep added descriptors
  _image_numbers.insert(std::make_pair(_train_descriptor_details.size(), image_number_));
  _train_descriptor_details.insert(std::make_pair(image_number_, std::set<int32_t>()));
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
  _total_duration_query_seconds += TOC(_time_begin).count();

  //ds clear bookkeeping to block N query descriptor to 1 train descriptor matches (manual crosscheck)
  for (std::pair<ImageNumberTrain, std::set<int32_t>> element: _train_descriptor_details) {
    element.second.clear();
  }

  //ds check for the best matching ratio over all neighbors
  std::multiset<ImageNumberTrain> number_of_matches_per_image;
  if (!k_matches_per_descriptor.empty()) {
    for (const std::vector<cv::DMatch>& matches: k_matches_per_descriptor) {
      for (const cv::DMatch& match: matches) {

        //ds readability
        const int32_t& descriptor_index_reference      = match.trainIdx;
        const ImageNumberTrain& image_number_reference = _image_numbers.at(match.imgIdx);
        std::set<int32_t>& matched_descriptors         = _train_descriptor_details.at(image_number_reference);

        //ds if the descriptor distance is acceptable and we didn't already match the descriptor for this image (cross check)
        if (match.distance < maximum_distance_hamming_                &&
            matched_descriptors.count(descriptor_index_reference) == 0) {

          //ds register descriptor match and store it (blocking further matching for this image)
          matched_descriptors.insert(descriptor_index_reference);
          number_of_matches_per_image.insert(image_number_reference);
        }
      }
    }

    //ds generate sortable score vector - based on all stored reference image numbers
    for (const std::pair<int32_t, ImageNumberTrain>& element: _image_numbers) {
      const ImageNumberTrain& image_number_reference = element.second;

      //ds if we can report the score for precision/recall evaluation
      if ((image_number_ >= _minimum_distance_between_closure_images                  &&
          image_number_reference <= image_number_-_minimum_distance_between_closure_images) ||
          _minimum_distance_between_closure_images == 0                                 ) {

        //ds compute relative matching score
        const double score = static_cast<double>(number_of_matches_per_image.count(image_number_reference))/query_descriptors_.rows;

        //ds add the closure
        closures_.push_back(ResultImageRetrieval(score, ImageNumberAssociation(image_number_, image_number_reference)));
      }
    }
  }

  //ds sort results in descending score
  std::sort(closures_.begin(), closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_)
      {return a_.number_of_matches_relative > b_.number_of_matches_relative;});
}
}
