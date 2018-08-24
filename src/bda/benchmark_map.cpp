#include <iostream>
#include "../matchers/bruteforce_matcher.h"
#include "../matchers/flannlsh_matcher.h"
#include "../utilities/command_line_parameters.h"

#ifdef SRRG_BENCH_BUILD_HBST
#include "../matchers/hbst_matcher.h"
#endif

#ifdef SRRG_BENCH_BUILD_DBOW2
#include "../matchers/bow_matcher.h"
#endif



int32_t main(int32_t argc_, char** argv_) {

  //ds validate number of parameters
  if (argc_ < 11) {
    std::cerr << "./benchmark_map -mode zubud"
                 "-images-reference <directory_train_images> "
                 "-images-query <directory_test_images> "
                 "-closures <ground_truth_mapping>.txt "
                 "-method <search_method>" << std::endl;
    return 0;
  }

  //ds disable multithreading of opencv
  cv::setNumThreads(0);
  cv::setUseOptimized(true);

  //ds grab configuration
  std::shared_ptr<srrg_bench::CommandLineParameters> parameters = std::make_shared<srrg_bench::CommandLineParameters>();
  parameters->parse(argc_, argv_);
  parameters->validate(std::cerr);

  //ds adjust thresholds
  parameters->maximum_descriptor_distance             = 0.1*DESCRIPTOR_SIZE_BITS;
  parameters->minimum_distance_between_closure_images = 1;
  parameters->maximum_leaf_size                       = parameters->target_number_of_descriptors;

  //ds configure and log
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds evaluated matcher
  std::shared_ptr<srrg_bench::BaseMatcher> matcher = 0;
  std::string method_name = parameters->method_name;

  //ds instantiate requested type
  if (method_name == "hbst") {
#ifdef SRRG_BENCH_BUILD_HBST
    srrg_bench::Tree::Node::maximum_leaf_size     = parameters->maximum_leaf_size;
    srrg_bench::Tree::Node::maximum_partitioning  = parameters->maximum_partitioning;
    matcher = std::make_shared<srrg_bench::HBSTMatcher>(parameters->minimum_distance_between_closure_images, srrg_hbst::SplitEven);
    method_name += "-"+std::to_string(srrg_bench::Tree::Node::maximum_leaf_size);
#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "bow") {
#ifdef SRRG_BENCH_BUILD_DBOW2
    matcher = std::make_shared<srrg_bench::BoWMatcher>(parameters->minimum_distance_between_closure_images,
                                           parameters->file_path_vocabulary,
                                           parameters->use_direct_index,
                                           parameters->direct_index_levels);

  //ds adjust descriptor type
  #if DBOW2_DESCRIPTOR_TYPE == 0
    parameters->descriptor_type = "brief";
  #elif DBOW2_DESCRIPTOR_TYPE == 1
    parameters->descriptor_type = "orb";
  #endif

#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "flannlsh") {
    matcher = std::make_shared<srrg_bench::FLANNLSHMatcher>(parameters->minimum_distance_between_closure_images,
                                                            parameters->table_number,
                                                            parameters->hash_key_size,
                                                            parameters->multi_probe_level);

    //ds store multi-probe level in name (0 indicates uniform LSH)
    method_name += ("-"+std::to_string(parameters->multi_probe_level));
  } else if (method_name == "bf") {
    matcher = std::make_shared<srrg_bench::BruteforceMatcher>(parameters->minimum_distance_between_closure_images, parameters->distance_norm);
  } else {
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
  }

  //ds create database - compute descriptors for each reference image
  std::cerr << "computing descriptors for reference images: " << std::endl;
  uint32_t number_of_processed_images    = 0;
  uint32_t number_of_trained_descriptors = 0;
  for (const srrg_bench::ImageWithPose* image_reference: parameters->evaluator->imagePosesGroundTruth()) {
    const cv::Mat image = cv::imread(image_reference->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    parameters->computeDescriptors(image, keypoints, descriptors);

    //ds add to database
    matcher->add(descriptors, number_of_processed_images, keypoints);

    //ds store descriptors
    std::cerr << "processed image: " << image_reference->file_name << " "
              << number_of_processed_images << "/" << parameters->evaluator->imagePosesGroundTruth().size()
              << " computed " << parameters->descriptor_type << " descriptors: " << keypoints.size() << std::endl;
    number_of_trained_descriptors += keypoints.size();
    ++number_of_processed_images;
  }

  //ds train database index
  std::cerr << "training index for "<< method_name << " with descriptors: " << number_of_trained_descriptors << std::endl;
  matcher->train();

  //ds evaluate each query
  std::cerr << "processing query images: " << std::endl;
  uint32_t number_of_processed_query_images = 0;
  for (const srrg_bench::ImageWithPose* image_query: parameters->evaluator->imagePosesQuery()) {
    const cv::Mat image = cv::imread(image_query->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    parameters->computeDescriptors(image, keypoints, descriptors);

    //ds query database
    std::vector<srrg_bench::ResultImageRetrieval> image_scores(0);
    matcher->query(descriptors, number_of_processed_images, parameters->maximum_descriptor_distance, image_scores);

    //ds scores
    ++number_of_processed_query_images;
    std::cerr << "processed image: " << image_query->file_name << " "
              << number_of_processed_query_images << "/" << parameters->evaluator->imagePosesQuery().size()
              << " computed " << parameters->descriptor_type << " descriptors: " << keypoints.size() << std::endl;
    for (uint32_t u = 0; u < 10; ++u) {
      const srrg_bench::ImageWithPose* image_reference = parameters->evaluator->imagePosesGroundTruth()[image_scores[u].image_association.train];
      std::cerr << image_query->file_name << " > " << image_reference->file_name
                << " : " << image_scores[u].number_of_matches_relative;
      if (parameters->evaluator->closureFeasibilityMap().at(image_query->image_number).count(image_reference->image_number)) {
        std::cerr << " MATCH (" << parameters->evaluator->closureFeasibilityMap().at(image_query->image_number).size() << ")" << std::endl;
      } else {
        std::cerr << std::endl;
      }
    }

    //ds display
    cv::Mat image_display = image;
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    cv::imshow("benchmark: current query image | ZuBuD", image_display);
    cv::waitKey(0);
  }

  return 0;
}
