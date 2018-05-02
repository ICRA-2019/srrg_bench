#pragma once
#include <chrono>
#include <ctime>
#include "../utilities/loop_closure_evaluator.h"

namespace srrg_bench {

//ds timinig readability
#define TIC(TIMER_) \
    TIMER_ = std::chrono::system_clock::now()
#define TOC(TIMER_) \
    static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now()-TIMER_)

//! @class base matcher interface for benchmarks
class BaseMatcher {

//ds object life
public:

  //! @brief default destructor
  virtual ~BaseMatcher() {};

//ds required interface
public:

  //! @brief database add function
  //! @param[in] train_descriptors_ collection of descriptors to be integrated into the database
  //! @param[in] image_number_ the image number associated with train_descriptors_
  //! @param[in] train_keypoints_ affiliated keypoints to the train_descriptors_ (optionally used)
  virtual void train(const cv::Mat& train_descriptors_,
                     const ImageNumberTrain& image_number_,
                     const std::vector<cv::KeyPoint>& train_keypoints_) = 0;

  //! @brief database matching function
  //! @param[in] query_descriptors_ collection of descriptors to match against the train_descriptors_ collections of each past image
  //! @param[in] image_number_ the image number associated with query_descriptors_
  //! @param[in] maximum_distance_hamming_ maximum allowed hamming distance for a valid match
  //! @param[out] closures_ collection of image to image matches
  virtual void query(const cv::Mat& query_descriptors_,
                     const ImageNumberQuery& image_number_,
                     const uint32_t& maximum_distance_hamming_,
                     std::vector<ResultImageRetrieval>& closures_) = 0;

  //! @brief resets the matcher and all structures - base method should be called from subclasses
  virtual void clear() {_durations_seconds_query_and_train.clear();}

  //! @brief duration for query and train operation
  const std::vector<double>& durationsSecondsQueryAndTrain() const {return _durations_seconds_query_and_train;}

  //! @brief number of simultaneous queries train operations
  const uint64_t numberOfQueries() const {return _durations_seconds_query_and_train.size();}

//ds attributes
protected:

  //! @brief matching and adding durations for each match/add call (continuous, not indexed)
  std::vector<double> _durations_seconds_query_and_train;

  //! @brief timing handle
  std::chrono::time_point<std::chrono::system_clock> _time_begin;
};
}
