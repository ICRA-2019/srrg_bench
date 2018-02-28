#include <chrono>
#include <ctime>
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
  bool vertical_matching_display = true;
  int32_t argc_parsed = 1;
  while(argc_parsed < argc_){
    if (!std::strcmp(argv_[argc_parsed], "-horizontal")) {
      vertical_matching_display = false;
    }
    argc_parsed++;
  }

  //ds allocate an empty tree
  std::shared_ptr<BinaryTree256> tree = std::make_shared<BinaryTree256>();

  //ds allocate a qt UI server in the main scope (required)
  std::shared_ptr<QApplication> ui_server = std::make_shared<QApplication>(argc_, argv_);

  //ds rotation for trajectory viewer
  Eigen::Isometry3d view(Eigen::Isometry3d::Identity());
  view.linear() << 0, -1, 0,
                   -1, 0, 0,
                   0, 0, -1;

  //ds viewer instance
  std::shared_ptr<srrg_bench::ViewerBonsai> viewer_tree  = std::make_shared<srrg_bench::ViewerBonsai>(tree);
  std::shared_ptr<srrg_bench::MapViewer> viewer_closures = std::make_shared<srrg_bench::MapViewer>();
  viewer_tree->show();
  viewer_tree->updateGL();
  viewer_closures->show();
  viewer_closures->updateGL();
  viewer_closures->setRotationRobotView(view);

  std::cerr << "===================================================================================================" << std::endl;
  std::cerr << "main|launched, press [Space] or [ARROW_UP] in GL window (title: BonsaiTree256) to start processing" << std::endl;
  std::cerr << "===================================================================================================" << std::endl;

//  //ds image dumping
//  uint64_t number_of_saved_images = 0;

  //ds current and previous image
  cv::Mat image_query;
  cv::Mat image_reference_best;
  std::vector<std::vector<cv::KeyPoint>> keypoints_per_image;

  //ds dataset variables
  uint64_t image_number_query = 0;
  const uint64_t& number_of_images = parameters->evaluator->numberOfImages();
  const uint32_t minimum_number_of_matches_for_display = 0.05*parameters->target_number_of_descriptors;

  //ds display GUI while active and we have images to process
  while (viewer_tree->isVisible()             &&
         viewer_closures->isVisible()         &&
         image_number_query < number_of_images) {

    //ds if stepwise playback is desired and no steps are set
    if(viewer_tree->optionStepwisePlayback()     &&
       viewer_tree->requestedPlaybackSteps() == 0) {
      viewer_closures->draw();
      viewer_closures->updateGL();
      viewer_tree->updateGL();
      ui_server->processEvents();
      continue;
    }

    //ds decrement steps if stepwise playback
    if (viewer_tree->optionStepwisePlayback()) {
      viewer_tree->decrementRequestedPlaybackSteps();
    }

    //ds load current image from disk
    image_query = cv::imread(parameters->evaluator->imagePosesGroundTruth()[image_number_query]->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds current pose
    Eigen::Isometry3d current_pose(parameters->evaluator->imagePosesGroundTruth()[image_number_query]->pose);

    //ds apply bayer decoding if necessary
    if (parameters->parsing_mode == "lucia" || parameters->parsing_mode == "oxford") {
      cv::cvtColor(image_query, image_query, CV_BayerGR2GRAY);
    }

    //ds detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    parameters->feature_detector->detect(image_query, keypoints);

    //ds sort keypoints descendingly by response value (for culling after descriptor computation)
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});

    //ds compute descriptors
    cv::Mat descriptors;
    parameters->descriptor_extractor->compute(image_query, keypoints, descriptors);

    //ds check insufficient descriptor number
    if (keypoints.size() < parameters->target_number_of_descriptors) {
      std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints.size()
                << " < " << parameters->target_number_of_descriptors << std::endl;
    }

    //ds rebuild descriptor matrix and keypoints vector
    keypoints.resize(std::min(keypoints.size(), static_cast<uint64_t>(parameters->target_number_of_descriptors)));
    descriptors = descriptors(cv::Rect(0, 0, descriptors.cols, keypoints.size()));
    keypoints_per_image.push_back(keypoints);

    //ds convert to keypoints linked to descriptors to HBST matchables
    BinaryTree256::MatchableVector current_matchables(BinaryTree256::getMatchablesWithPointer<cv::KeyPoint>(descriptors, keypoints_per_image.back(), image_number_query));

    //ds match map for this query image
    BinaryTree256::MatchVectorMap matches;

    //ds current reference poses
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> reference_poses(0);

    //ds if we are in query range
    if (image_number_query > parameters->minimum_distance_between_closure_images) {

      //ds reference image number for which we got the most matches
      uint64_t image_number_reference = 0;

      //ds query the tree for a similar matchable vector (e.g. an image)
      std::chrono::time_point<std::chrono::system_clock> time_begin = std::chrono::system_clock::now();
      tree->match(current_matchables, matches);
      std::chrono::duration<double> duration_seconds_query = std::chrono::system_clock::now()-time_begin;

      //ds check for the best matching ratio in the result
      uint32_t best_matches = 0;
      for (const BinaryTree256::MatchVectorMapElement& match_vector: matches) {

        //ds if in chronological order and not to recent
        if (match_vector.first <= image_number_query-parameters->minimum_distance_between_closure_images) {

          //ds if the threshold is fine
          if (match_vector.second.size() > minimum_number_of_matches_for_display) {
            reference_poses.push_back(parameters->evaluator->imagePosesGroundTruth()[match_vector.first]->pose);
          }

          //ds if we got a better score
          if (match_vector.second.size() > best_matches) {
            image_number_reference = match_vector.first;
            best_matches           = match_vector.second.size();
          }
        }
      }

      //ds quantitative
      std::printf("main|process best: [%08lu] > [%08lu] with matches: %4u (QUERY size: %5lu duration (s): %6.4f)\n",
                  image_number_query,
                  image_number_reference,
                  best_matches,
                  current_matchables.size(),
                  duration_seconds_query.count());

      //ds update query display
      viewer_tree->setMatches(matches[image_number_reference]);
      image_reference_best = cv::imread(parameters->evaluator->imagePosesGroundTruth()[image_number_reference]->file_name, CV_LOAD_IMAGE_GRAYSCALE);
      if (parameters->parsing_mode == "lucia" || parameters->parsing_mode == "oxford") {
        cv::cvtColor(image_reference_best, image_reference_best, CV_BayerGR2GRAY);
      }
    } else {

      //ds info
      std::printf("main|process [%08lu] not yet in query range for interspace: %u\n",
                  image_number_query,
                  parameters->minimum_distance_between_closure_images);
    }

    //ds train tree with current matchables
    tree->add(current_matchables);

    //ds display image
    cv::Mat image_display;

    //ds if there are no matches available
    if (viewer_tree->matches().empty()) {

      //ds display detected points
      cv::Mat empty_image(image_query.clone());
      empty_image.setTo(0);
      if (vertical_matching_display) {
        cv::vconcat(image_query, empty_image, image_display);
      } else {
        cv::hconcat(image_query, empty_image, image_display);
      }
      cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
      for (const BinaryTree256::Matchable* matchable: current_matchables) {
        const cv::KeyPoint& keypoint = *(reinterpret_cast<const cv::KeyPoint*>(matchable->pointer));
        cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0));
      }
    } else {

      //ds show query image on top and reference on bottom
      cv::Point2f shift(0, 0);
      if (vertical_matching_display) {
        cv::vconcat(image_query, image_reference_best, image_display);
        shift = cv::Point2f(0, image_reference_best.rows);
      } else {
        cv::hconcat(image_query, image_reference_best, image_display);
        shift = cv::Point2f(image_reference_best.cols, 0);
      }
      cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

      //ds color mode depending on number of matches
      cv::Scalar color_correspondence(0, 255, 0);
      if (viewer_tree->matches().size() < minimum_number_of_matches_for_display) {
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
    viewer_closures->setRobotToWorld(current_pose);
    viewer_closures->setReferencePoses(reference_poses);

    //ds update opencv window
    cv::imshow("Descriptor Matching (top: QUERY, bot: REFERENCE)", image_display);
    cv::waitKey(1);

//    //ds save whole image to disk
//    char buffer_file_name[32];
//    std::snprintf(buffer_file_name, 32, "images/input-%04lu.jpg", number_of_saved_images);

    //ds update GL
    viewer_closures->draw();
    viewer_closures->updateGL();
    viewer_tree->updateGL();
    ui_server->processEvents();

//    //ds safe images if new
//    if (have_new_image) {
//      viewer_closures->setSnapshotFileName("images/trajectory.jpg");
//      viewer_closures->saveSnapshot();
//      viewer_tree->setSnapshotFileName("images/tree.jpg");
//      viewer_tree->saveSnapshot();
//      ++number_of_saved_images;
//    }

    //ds process next image
    ++image_number_query;
  }

  //ds close cv windows
  cv::destroyAllWindows();
  ui_server->closeAllWindows();
  ui_server->quit();
  return 0;
}
