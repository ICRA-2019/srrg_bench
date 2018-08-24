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
  parameters->minimum_distance_between_closure_images = 0;
  parameters->maximum_leaf_size                       = parameters->target_number_of_descriptors;

  //ds configure and log
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds evaluated matcher
  std::shared_ptr<srrg_bench::BaseMatcher> matcher            = 0;
  std::string method_name                                     = parameters->method_name;
  std::shared_ptr<srrg_bench::LoopClosureEvaluator> evaluator = parameters->evaluator;

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
  uint32_t number_of_processed_reference_images = 0;
  uint32_t number_of_trained_descriptors        = 0;
  for (const srrg_bench::ImageWithPose* image_reference: evaluator->imagePosesGroundTruth()) {
    const cv::Mat image = cv::imread(image_reference->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    parameters->computeDescriptors(image, keypoints, descriptors);

    //ds add to database
    matcher->add(descriptors, number_of_processed_reference_images, keypoints);

    //ds store descriptors
    std::cerr << "processed REFERENCE image: '" << image_reference->file_name
              << "' " << number_of_processed_reference_images << "/" << evaluator->imagePosesGroundTruth().size()
              << " (" << image_reference->image_number << ")"
              << " computed <" << parameters->descriptor_type << "> descriptors: " << keypoints.size() << std::endl;
    number_of_trained_descriptors += keypoints.size();
    ++number_of_processed_reference_images;
  }

  //ds train database index
  std::cerr << "training index for <"<< method_name << "> with descriptors: " << number_of_trained_descriptors << std::endl;
  matcher->train();

  //ds evaluate each query
  std::cerr << "processing query images: " << std::endl;
  uint32_t number_of_processed_query_images = 0;
  double mean_average_precision = 0;
  std::chrono::time_point<std::chrono::system_clock> timer;
  for (const srrg_bench::ImageWithPose* image_query: evaluator->imagePosesQuery()) {
    const cv::Mat image = cv::imread(image_query->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    parameters->computeDescriptors(image, keypoints, descriptors);

    //ds query database
    std::vector<srrg_bench::ResultImageRetrieval> image_scores(0);
    matcher->query(descriptors, image_query->image_number, parameters->maximum_descriptor_distance, image_scores);

    //ds adjust reference image numbers to actual dataset image numbers (since we have duplicates with zubud)
    if (parameters->parsing_mode == "zubud") {
      for (srrg_bench::ResultImageRetrieval& image_score: image_scores) {
        image_score.image_association.train = evaluator->imagePosesGroundTruth()[image_score.image_association.train]->image_number;
      }
    }

    //ds compute average precision
    const std::multiset<srrg_bench::ImageNumberTrain> valid_reference_image_list(evaluator->closureFeasibilityMap().at(image_query->image_number));
    const double average_precision = evaluator->computeAveragePrecision(image_scores, valid_reference_image_list);
    mean_average_precision        += average_precision;
    ++number_of_processed_query_images;
    std::cerr << "processed QUERY image: '" << image_query->file_name
              << "' " << number_of_processed_query_images << "/" << evaluator->imagePosesQuery().size()
              << " (" << image_query->image_number << ")"
              << " computed <" << parameters->descriptor_type << "> descriptors: " << keypoints.size()
              << " | AP: " << average_precision << std::endl;

    //ds display
    cv::Mat image_display = image;
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
    for (const cv::KeyPoint& keypoint: keypoints) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    cv::imshow("benchmark: current query image | "+parameters->parsing_mode, image_display);
    cv::waitKey(1);
  }
  std::cerr << BAR << std::endl;

  //ds compute mAP
  mean_average_precision /= number_of_processed_query_images;
  std::cerr << "summary for <" << parameters->parsing_mode << "><" << parameters->descriptor_type << "><" << method_name << ">" << std::endl;
  std::cerr << BAR << std::endl;
  std::cerr << "number of processed reference images: " << number_of_processed_reference_images << std::endl;
  std::cerr << "    number of processed query images: " << number_of_processed_query_images << std::endl;
  std::cerr << "        mean average precision (mAP): " << mean_average_precision << std::endl;
  std::cerr << "        mean add processing time (s): " << matcher->totalDurationAddSeconds()/number_of_processed_reference_images << std::endl;
  std::cerr << "           train processing time (s): " << matcher->totalDurationTrainSeconds() << std::endl;
  std::cerr << "      mean query processing time (s): " << matcher->totalDurationQuerySeconds()/number_of_processed_query_images << std::endl;
  return 0;
}
