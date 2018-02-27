#include <chrono>
#include <ctime>
#include <thread>
#include <atomic>
#include <mutex>
#include "qapplication.h"
#include "srrg_hbst_types/binary_tree.hpp"
#include "visualization/viewer_bonsai.h"
#include "visualization/map_viewer.h"
#include "utilities/command_line_parameters.h"

#if CV_MAJOR_VERSION == 2
  //ds no specifics
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

using namespace srrg_hbst;

class BonsaiAssembly {

//ds object management
public:

  BonsaiAssembly(): images_available_for_display(false),
                    _is_running(false),
                    _is_waiting(false),
                    _termination_requested(false),
                    _requested_playback_steps(0),
                    _updated_pose(false) {
    _images.clear();
    _buffer_keypoints.clear();
    _matchables_last.clear();
    _mutex_data_transfer = new std::mutex();
    _image_numbers_match_last = std::make_pair(0, 0);
  }
  ~BonsaiAssembly() {}

//ds access
public:

  //ds process provided sequence
  void process(const std::vector<srrg_bench::ImageWithPose*>& images_, const uint64_t& minimum_distance_between_closure_images_) {

    //ds for each specified image
    for (uint64_t image_number = 0; image_number < images_.size(); ++image_number) {

      //ds check for termination
      if (_termination_requested) {
        return;
      }

      //ds check for wait request
      while (_is_waiting) {

        //ds check for termination while waiting
        if (_termination_requested) {
          return;
        }

        //ds check for stepping request
        if (_requested_playback_steps > 0) {
          break;
        }

        //ds breath
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      //ds if we're stepping
      if (_requested_playback_steps > 0) {
        --_requested_playback_steps;
      }

      //ds load image from disk
      cv::Mat image = cv::imread(images_[image_number]->file_name, CV_LOAD_IMAGE_GRAYSCALE);

      //ds apply bayer decoding if necessary
      if (_parameters->parsing_mode == "lucia" || _parameters->parsing_mode == "oxford") {
        cv::cvtColor(image, image, CV_BayerGR2GRAY);
      }
      _images.insert(std::make_pair(image_number, image));

      //ds detect FAST keypoints
      std::vector<cv::KeyPoint> keypoints;
      _parameters->feature_detector->detect(image, keypoints);

      //ds sort keypoints descendingly by response value (for culling after descriptor computation)
      std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});

      //ds compute descriptors
      cv::Mat descriptors;
      _parameters->descriptor_extractor->compute(image, keypoints, descriptors);

      //ds check insufficient descriptor number
      if (keypoints.size() < _parameters->target_number_of_descriptors) {
        std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints.size()
                  << " < " << _parameters->target_number_of_descriptors << std::endl;
      }

      //ds rebuild descriptor matrix and keypoints vector
      keypoints.resize(std::min(keypoints.size(), static_cast<uint64_t>(_parameters->target_number_of_descriptors)));
      descriptors = descriptors(cv::Rect(0, 0, descriptors.cols, keypoints.size()));

      //ds store final keypoints
      _buffer_keypoints.push_back(keypoints);

      //ds convert to keypoints linked to descriptors to HBST matchables
      BinaryTree256::MatchableVector matchables(BinaryTree256::getMatchablesWithPointer<cv::KeyPoint>(descriptors, _buffer_keypoints.back(), image_number));

      //ds reference poses (based on matching threshold)
      _reference_poses.clear();

      //ds if we have a tree
      if (_tree) {

        //ds match map for this query image
        BinaryTree256::MatchVectorMap matches;

        //ds reference image number for which we got the most matches
        uint64_t image_number_reference = 0;

        //ds if we are in query range
        if (image_number > minimum_distance_between_closure_images_) {

          //ds query the tree for a similar matchable vector (e.g. an image)
          _time_begin = std::chrono::system_clock::now();
          _tree->match(matchables, matches);
          _duration_seconds_query = std::chrono::system_clock::now()-_time_begin;

          //ds check for the best matching ratio in the result
          uint32_t best_matches = 0;
          for (const BinaryTree256::MatchVectorMapElement& match_vector: matches) {

            //ds if in chronological order and not to recent
            if (match_vector.first <= image_number-minimum_distance_between_closure_images_) {

              //ds if the threshold is fine
              if (match_vector.second.size() > 150) {
                _reference_poses.push_back(images_[match_vector.first]->pose);
              }

              //ds if we got a better score
              if (match_vector.second.size() > best_matches) {
                image_number_reference = match_vector.first;
                best_matches           = match_vector.second.size();
              }
            }
          }

          //ds quantitative
          std::printf("BonsaiAssembly::process|best: [%08lu] > [%08lu] with matches: %4u (QUERY size: %5lu duration (s): %6.4f)\n",
                      image_number,
                      image_number_reference,
                      best_matches,
                      matchables.size(),
                      _duration_seconds_query.count());
        } else {

          //ds info
          std::printf("BonsaiAssembly::process|[%08lu] not yet in query range for interspace: %lu\n", image_number, minimum_distance_between_closure_images_);
        }

        //ds wait until available and lock drawing in viewer
        std::lock_guard<std::mutex> lock(*_mutex_data_transfer);

        //ds update last handles
        _image_numbers_match_last = std::make_pair(image_number, image_number_reference);
        _matchables_last.clear();
        _matchables_last.insert(_matchables_last.end(), matchables.begin(), matchables.end());
        images_available_for_display = true;

        //ds train tree with current matchables
        _tree->add(matchables, _train_mode);

        //ds if we got a viewer instance to synchronize with and we computed some matches
        if (_viewer && !matches.empty()) {

          //ds update query display
          _viewer->setMatches(matches[image_number_reference]);

          //ds if a break is desired
          if (_viewer->optionStepwisePlayback()) {

            //ds trigger inner waiting loop
            _is_waiting = true;
          }
        }
      } else {

        //ds wait until available and lock drawing in viewer
        std::lock_guard<std::mutex> lock(*_mutex_data_transfer);

        //ds set match info
        _image_numbers_match_last = std::make_pair(image_number, 0);
        _matchables_last.clear();
        _matchables_last.insert(_matchables_last.end(), matchables.begin(), matchables.end());
        images_available_for_display = true;

        //ds build a new tree
        _tree = new BinaryTree256(image_number, matchables, _train_mode);

        //ds if we got a viewer instance to synchronize with
        if (_viewer) {

          //ds set new tree to viewer
          _viewer->setTree(_tree);

          //ds if a break is desired
          if (_viewer->optionStepwisePlayback()) {

            //ds trigger inner waiting loop
            _is_waiting = true;
          }
        }
      }

      //ds update current pose
      _current_pose = images_[image_number]->pose;
      _updated_pose = true;
    }
    _is_running = false;
  }

  //ds thread wrapping
  std::shared_ptr<std::thread> processInThread(const std::vector<srrg_bench::ImageWithPose*>& images_, const uint64_t& minimum_distance_between_closure_images_) {
    _is_running = true;
    return std::make_shared<std::thread>([=] {process(images_, minimum_distance_between_closure_images_);});
  }

  //ds clear processing
  void clear() {
    _images.clear();
    _buffer_keypoints.clear();
    _matchables_last.clear();
    if (_tree) {delete _tree;}
    if (_mutex_data_transfer) {delete _mutex_data_transfer;}
  }

  //ds request thread termination
  void requestTermination(const std::string& reason_) {
    std::cerr << "BonsaiAssembly::requestTermination|termination requested - '" << reason_ << "'" << std::endl;
    _termination_requested = true;
  }

  //ds check running status
  inline const bool isRunning() const {return _is_running;}
  inline const bool isWaiting() const {return _is_waiting;}
  inline void setIsWaiting(const bool& is_waiting_) {_is_waiting = is_waiting_;}
  inline void setRequestedPlaybackSteps(const uint32_t& requested_playback_steps_) {_requested_playback_steps += requested_playback_steps_;}

  //ds setter for viewer instance
  void setViewer(std::shared_ptr<srrg_bench::ViewerBonsai> viewer_) {_viewer = viewer_;}

  //ds external mutex access for modules interacting with this
  std::mutex* mutexDataTransfer() {return _mutex_data_transfer;}

  //ds drawing access
  const cv::Mat imageQueryLast() const {return _images.at(_image_numbers_match_last.first);}
  const cv::Mat imageReferenceLast() const {return _images.at(_image_numbers_match_last.second);}
  const BinaryTree256::MatchableVector matchablesLast() const {return _matchables_last;}
  const Eigen::Isometry3d& currentPose() {_updated_pose = false; return _current_pose;}
  inline const bool updatedPose() const {return _updated_pose;}
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& referencePoses() const {return _reference_poses;}
  void setTrainMode(const SplittingStrategy& train_mode_) {_train_mode = train_mode_;}

  void setParameters(std::shared_ptr<srrg_bench::CommandLineParameters> parameters_) {_parameters = parameters_;}

//ds interaction
public:

  std::atomic<bool> images_available_for_display;

//ds attributes
private:

  //ds thread control
  std::atomic<bool> _is_running;
  std::atomic<bool> _is_waiting;
  std::atomic<bool> _termination_requested;
  std::mutex* _mutex_data_transfer;
  std::atomic<uint32_t> _requested_playback_steps;

  //ds parameters
  std::shared_ptr<srrg_bench::CommandLineParameters> _parameters;

  //ds global buffers - otherwise the memory pointers get broken by the scope
  std::map<uint64_t, cv::Mat> _images;
  std::vector<std::vector<cv::KeyPoint>> _buffer_keypoints;
  std::pair<uint64_t, uint64_t> _image_numbers_match_last;
  BinaryTree256::MatchableVector _matchables_last;
  std::atomic<bool> _updated_pose;
  Eigen::Isometry3d _current_pose;
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> _reference_poses;

  //ds active tree handle
  BinaryTree256* _tree = 0;

  //ds train mode
  SplittingStrategy _train_mode = SplittingStrategy::SplitEven;

  //ds reference GUI handle
  std::shared_ptr<srrg_bench::ViewerBonsai> _viewer = 0;

  //ds timinig
  std::chrono::time_point<std::chrono::system_clock> _time_begin;
  std::chrono::duration<double> _duration_seconds_query;
};

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 4) {
    std::cerr << "EXPERIMENTAL" << std::endl;
    return 0;
  }

  //ds grab configuration
  std::shared_ptr<srrg_bench::CommandLineParameters> parameters = std::make_shared<srrg_bench::CommandLineParameters>();
  parameters->parse(argc_, argv_);
  parameters->validate(std::cerr);
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds display configuration
  bool vertical_matching_display = false;

  //ds update hbst parameters
  BinaryTree256::Node::maximum_leaf_size    = 50;
  BinaryTree256::Node::maximum_partitioning = 0.1;

  //ds allocate a bonsai assembly
  std::shared_ptr<BonsaiAssembly> bonsai_assembly = std::make_shared<BonsaiAssembly>();
  bonsai_assembly->setParameters(parameters);

  //ds allocate a qt UI server in the main scope (required)
  QApplication* ui_server = new QApplication(argc_, argv_);

  //ds rotation for trajectory viewer
  Eigen::Isometry3d view(Eigen::Isometry3d::Identity());
  view.linear() << 0, -1, 0,
                   -1, 0, 0,
                   0, 0, -1;

  //ds viewer instance
  std::shared_ptr<srrg_bench::ViewerBonsai> viewer_tree = std::make_shared<srrg_bench::ViewerBonsai>();
  std::shared_ptr<srrg_bench::MapViewer> viewer_closures = std::make_shared<srrg_bench::MapViewer>();
  viewer_closures->show();
  viewer_closures->updateGL();
  viewer_closures->setRotationRobotView(view);
  viewer_tree->configure(bonsai_assembly->mutexDataTransfer());
  viewer_tree->show();
  viewer_tree->updateGL();
  bonsai_assembly->setViewer(viewer_tree);

  //ds start HBST matching in a thread
  std::shared_ptr<std::thread> process_thread = bonsai_assembly->processInThread(parameters->evaluator->imagePosesGroundTruth(),
                                                                                 parameters->minimum_distance_between_closure_images);
  std::cerr << "===================================================================================================" << std::endl;
  std::cerr << "main|launched, press [Space] or [ARROW_UP] in GL window (title: BonsaiTree256) to start processing" << std::endl;
  std::cerr << "===================================================================================================" << std::endl;

  //ds opencv image to display
  cv::Mat image_display;
//  uint64_t number_of_saved_images = 0;

  //ds display GUI
  while (viewer_tree->isVisible() && viewer_closures->isVisible()) {
    bool have_new_image = bonsai_assembly->images_available_for_display;
    bonsai_assembly->images_available_for_display = false;

    //ds if there are images available
    if (have_new_image) {

      //ds if there are no matches available
      if (viewer_tree->matches().empty()) {

        //ds display detected points
        cv::Mat empty_image(bonsai_assembly->imageQueryLast().clone());
        empty_image.setTo(0);
        if (vertical_matching_display) {
          cv::vconcat(bonsai_assembly->imageQueryLast(), empty_image, image_display);
        } else {
          cv::hconcat(bonsai_assembly->imageQueryLast(), empty_image, image_display);
        }
        cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
        for (const BinaryTree256::Matchable* matchable: bonsai_assembly->matchablesLast()) {
          const cv::KeyPoint& keypoint = *(reinterpret_cast<const cv::KeyPoint*>(matchable->pointer));
          cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0));
        }
      } else {

        //ds show query image on top and reference on bottom
        cv::Point2f shift(0, 0);
        if (vertical_matching_display) {
          cv::vconcat(bonsai_assembly->imageQueryLast(), bonsai_assembly->imageReferenceLast(), image_display);
          shift = cv::Point2f(0, bonsai_assembly->imageReferenceLast().rows);
        } else {
          cv::hconcat(bonsai_assembly->imageQueryLast(), bonsai_assembly->imageReferenceLast(), image_display);
          shift = cv::Point2f(bonsai_assembly->imageReferenceLast().cols, 0);
        }
        cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

        //ds color mode depending on number of matches
        cv::Scalar color_correspondence(0, 255, 0);
        if (viewer_tree->matches().size() < 100) {
          color_correspondence = cv::Scalar(200, 200, 200);
        }

        //ds draw correspondences in opencv image
        for (const BinaryTree256::Match& match: viewer_tree->matches()) {

          //ds directly get the keypoint objects
          const cv::KeyPoint& keypoint_reference = *(reinterpret_cast<const cv::KeyPoint*>(match.pointer_reference));
          const cv::KeyPoint& keypoint_query     = *(reinterpret_cast<const cv::KeyPoint*>(match.pointer_query));

          //ds draw correspondence
          cv::line(image_display, keypoint_query.pt, keypoint_reference.pt+shift, color_correspondence);

          //ds draw query point in upper image
          cv::circle(image_display, keypoint_query.pt, 2, cv::Scalar(255, 0, 0));

          //ds draw reference point in lower image
          cv::circle(image_display, keypoint_reference.pt+shift, 2, cv::Scalar(0, 0, 255));
        }
      }

      //ds update trajectory display
      if (bonsai_assembly->updatedPose()) {
        viewer_closures->setRobotToWorld(bonsai_assembly->currentPose());
      }
      viewer_closures->setReferencePoses(bonsai_assembly->referencePoses());

      //ds update opencv window
      cv::imshow("Descriptor Matching (top: QUERY, bot: REFERENCE)", image_display);
      cv::waitKey(1);

//      //ds save whole image to disk
//      char buffer_file_name[32];
//      std::snprintf(buffer_file_name, 32, "images/input-%04lu.jpg", number_of_saved_images);
//      cv::imwrite(buffer_file_name, image_display);
    }

    //ds breathe
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    //ds update GL
    viewer_closures->draw();
    viewer_closures->updateGL();
    viewer_tree->updateGL();
    ui_server->processEvents();

    //ds safe images if new
//    if (have_new_image) {
//      viewer_closures->setSnapshotFileName("images/trajectory.jpg");
//      viewer_closures->saveSnapshot();
//      viewer_tree->setSnapshotFileName("images/tree.jpg");
//      viewer_tree->saveSnapshot();
//      ++number_of_saved_images;
//    }

    //ds first check if we got a new option request (avoid unnecessary locking)
    if (viewer_tree->optionRequested()) {

      //ds check if we have to lock the playback
      if (viewer_tree->optionStepwisePlayback()) {

        //ds if its not a stepping request
        if (viewer_tree->requestedPlaybackSteps() == 0) {
          std::cerr << "main|switched to stepwise mode, press [ARROW_UP] to step" << std::endl;
        }
        bonsai_assembly->setIsWaiting(true);
        bonsai_assembly->setRequestedPlaybackSteps(viewer_tree->requestedPlaybackSteps());
        viewer_tree->resetRequestedPlaybackSteps();
      } else {
        std::cerr << "main|switched to benchmark mode" << std::endl;
        viewer_tree->resetRequestedPlaybackSteps();
        bonsai_assembly->setIsWaiting(false);
      }
      viewer_tree->setOptionRequested(false);
    }
  }

  //ds close cv windows
  cv::destroyAllWindows();
  ui_server->closeAllWindows();
  ui_server->quit();
  delete ui_server;

  //ds signal threads
  bonsai_assembly->requestTermination("user closed GUI");

  //ds make sure all threads are terminated
  std::cerr << "main|joining thread: process" << std::endl;
  process_thread->join();

  //ds done
  std::cerr << "main|freeing memory .." << std::endl;
  std::cerr << "main|done" << std::endl;
  return 0;
}
