#include "matchers/bruteforce_matcher.h"
#include "matchers/flannlsh_matcher.h"
#include "utilities/command_line_parameters.h"

#ifdef SRRG_BENCH_BUILD_HBST
#include "matchers/hbst_matcher.h"
#endif

#ifdef SRRG_BENCH_BUILD_DBOW2
#include "matchers/bow_matcher.h"
#endif

#ifdef SRRG_BENCH_BUILD_IBOW
#include "matchers/ibow_matcher.h"
#endif

#ifdef SRRG_BENCH_BUILD_FLANNHC
#include "matchers/flannhc_matcher.h"
#endif

#ifdef SRRG_BENCH_BUILD_VIEWERS
#include <thread>
#include "qapplication.h"
#include "visualization/closure_viewer.h"
#endif

using namespace srrg_bench;

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 9) {
    std::cerr << "use: ./benchmark [-cross] -method <hbst/bow/flannlsh/flannhc> -mode <kitti/malaga/lucia/oxford> -images <folder_images_0> [<folder_images_1>] -poses <poses_gt_0> [<poses_gt_1>]"
                 "[-closures <closures_gt> -r <target_recall> -t <fast_threshold> -n <target_number_of_descriptors> -space <image_number_interspace> "
                 "-descriptor <brief/orb/akaze/brisk> -voc <file_descriptor_vocabulary> -use-gui (-ug) -score-only (-so)]" << std::endl;
    return EXIT_FAILURE;
  }

  //ds disable multithreading of opencv
  cv::setNumThreads(0);

  //ds grab configuration
  std::shared_ptr<CommandLineParameters> parameters = std::make_shared<CommandLineParameters>();
  parameters->parse(argc_, argv_);
  parameters->validate(std::cerr);
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds evaluated matcher
  std::shared_ptr<BaseMatcher> matcher = 0;
  std::string method_name = parameters->method_name;

  //ds instantiate requested type
  if (method_name == "hbst") {
#ifdef SRRG_BENCH_BUILD_HBST
    Tree::Node::maximum_leaf_size     = parameters->maximum_leaf_size;
    Tree::Node::maximum_partitioning  = parameters->maximum_partitioning;
    if (parameters->use_random_splitting) {
      matcher = std::make_shared<HBSTMatcher>(parameters->query_interspace,
                                              parameters->minimum_distance_between_closure_images,
                                              srrg_hbst::SplittingStrategy::SplitRandomUniform);
      method_name += "-random-" + std::to_string(Tree::Node::maximum_leaf_size);
    } else if (parameters->use_uneven_splitting) {
      matcher = std::make_shared<HBSTMatcher>(parameters->query_interspace,
                                              parameters->minimum_distance_between_closure_images,
                                              srrg_hbst::SplittingStrategy::SplitUneven);
      method_name += "-uneven-" + std::to_string(Tree::Node::maximum_leaf_size);
    } else {
      matcher = std::make_shared<HBSTMatcher>(parameters->query_interspace,
                                              parameters->minimum_distance_between_closure_images,
                                              srrg_hbst::SplittingStrategy::SplitEven);
      method_name += "-even-" + std::to_string(Tree::Node::maximum_leaf_size);
    }
#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "bow") {
#ifdef SRRG_BENCH_BUILD_DBOW2
    matcher = std::make_shared<BoWMatcher>(parameters->query_interspace,
                                           parameters->minimum_distance_between_closure_images,
                                           parameters->file_path_vocabulary,
                                           parameters->use_direct_index,
                                           parameters->direct_index_levels,
                                           parameters->compute_score_only);
    if (parameters->compute_score_only) {
      method_name += "-so";
    }
#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "ibow") {
#ifdef SRRG_BENCH_BUILD_IBOW
    matcher = std::make_shared<iBoWMatcher>(parameters->query_interspace,
                                            parameters->minimum_distance_between_closure_images);
    if (parameters->compute_score_only) {
      method_name += "-so";
    }
#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "flannlsh") {
    matcher = std::make_shared<FLANNLSHMatcher>(parameters->query_interspace,
                                                parameters->minimum_distance_between_closure_images,
                                                parameters->table_number,
                                                parameters->key_size,
                                                parameters->multi_probe_level);

    //ds enable optimization
    cv::setUseOptimized(true);
  } else if (method_name == "flannhc") {
#ifdef SRRG_BENCH_BUILD_FLANNHC
    matcher = std::make_shared<FLANNHCMatcher>(parameters->query_interspace,
                                               parameters->minimum_distance_between_closure_images);
#else
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
#endif
  } else if (method_name == "bf") {
    matcher = std::make_shared<BruteforceMatcher>(parameters->query_interspace,
                                                  parameters->minimum_distance_between_closure_images);

    //ds enable optimization and multithreading
    cv::setNumThreads(4);
    cv::setUseOptimized(true);
  } else {
    std::cerr << "ERROR: unknown method name: " << method_name << std::endl;
    return EXIT_FAILURE;
  }

  //ds compute outfile suffix
  const std::string benchmark_suffix = method_name + "_"
                                     + parameters->descriptor_type + "_"
                                     + std::to_string(parameters->target_number_of_descriptors) + ".txt";
  LOG_VARIABLE(benchmark_suffix)
  LOG_VARIABLE(cv::getNumThreads())

  //ds output recall: precision(s) curve for evaluated samples
  std::vector<double> maximum_f1_scores(0);

  //ds average runtime (over all samples)
  std::vector<double> mean_processing_times(0);

  //ds for the number of test runs
  for (uint32_t sample_index = 0; sample_index < parameters->number_of_samples; ++sample_index) {
    matcher->clear();

    //ds result vector (evaluated for precision/recall)
    std::vector<ResultImageRetrieval> closures(0);

    //ds start benchmark
    const uint32_t length_progress_bar   = 50;
    uint64_t number_of_descriptors_total = 0;
    std::vector<uint64_t> query_identifiers;
    std::vector<uint64_t> number_of_descriptors_accumulated(0);
    for (ImageNumberQuery image_number_query = 0; image_number_query < parameters->evaluator->numberOfImages(); ++image_number_query) {

      //ds if we got a query image
      if (image_number_query%parameters->query_interspace == 0) {

        //ds load image from disk
        const std::string file_name_image = parameters->evaluator->imagePosesGroundTruth()[image_number_query]->file_name;
        cv::Mat image = cv::imread(file_name_image, CV_LOAD_IMAGE_GRAYSCALE);

        //ds apply bayer decoding if necessary
        if (parameters->parsing_mode == "lucia" || parameters->parsing_mode == "oxford") {
          cv::cvtColor(image, image, CV_BayerGR2GRAY);
        }

        //ds detect FAST keypoints
        std::vector<cv::KeyPoint> keypoints;
        parameters->feature_detector->detect(image, keypoints);

        //ds sort keypoints descendingly by response value (for culling after descriptor computation)
        std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});

        //ds compute descriptors
        cv::Mat descriptors;
        parameters->descriptor_extractor->compute(image, keypoints, descriptors);

        //ds check insufficient descriptor number
        if (keypoints.size() < parameters->target_number_of_descriptors) {
          std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints.size()
                    << " < " << parameters->target_number_of_descriptors << ", adjust detector threshold (skipping this image: " << image_number_query << ")" << std::endl;
          continue;
        }

        //ds rebuild descriptor matrix and keypoints vector
        keypoints.resize(parameters->target_number_of_descriptors);
        descriptors = descriptors(cv::Rect(0, 0, descriptors.cols, parameters->target_number_of_descriptors));

        //ds convert to points
        std::vector<cv::Point2f> points_current(parameters->target_number_of_descriptors);
        for (uint64_t u = 0; u < keypoints.size(); ++u) {
          points_current[u] = keypoints[u].pt;
        }

        //ds if desired - draw keypoints with descriptors
        cv::Mat image_display;
        const cv::Point2f offset(0, image.rows);
        if (parameters->use_gui) {

          //ds build image window
          image_display = image;
          cv::cvtColor(image_display, image_display, CV_GRAY2RGB);

          //ds draw current and previous keypoints
          for (const cv::Point2f& point: points_current) {
            cv::circle(image_display, point, 2, cv::Scalar(255, 0, 0), -1);
          }
        }

        //ds update statistics
        number_of_descriptors_total += parameters->target_number_of_descriptors;
        number_of_descriptors_accumulated.push_back(number_of_descriptors_total);

        //ds display currently detected keypoints
        if (parameters->use_gui) {
          cv::imshow("benchmark: current image | " + benchmark_suffix, image_display);
          cv::waitKey(1);
        }

        //ds query against all past images, retrieving closures with relative scores
        matcher->query(descriptors, image_number_query, parameters->maximum_distance_hamming, closures);

        //ds train on the query image
        matcher->train(descriptors, image_number_query, keypoints);

        //free memory
        image.release();
      }

      //ds progress feedback: compute printing configuration
      const double progress           = static_cast<double>(image_number_query+1)/parameters->evaluator->numberOfImages();
      const uint32_t length_completed = progress*length_progress_bar;

      //ds draw progress bar
      std::cerr << "progress [";
      for (uint32_t u = 0; u < length_completed; ++u) {
        std::cerr << "+";
      }
      for (uint32_t u = length_completed; u < length_progress_bar; ++u) {
        std::cerr << " ";
      }
      std::cerr << "] " << static_cast<int32_t>(progress*100.0) << " %"
                        << " images: " << image_number_query+1
                        << " queries: " << number_of_descriptors_accumulated.size()
                        << " descriptors: " << number_of_descriptors_total
                        << " (average: " << static_cast<double>(number_of_descriptors_total)/(image_number_query+1) << ")"
                        << " sample: " << sample_index
                        << " current processing time (s): " << matcher->durationsSecondsQueryAndTrain().back() << "\r";
    }

    //ds retrieve precision/recall for current result and save it to a file
    double maximum_f1_score = 0;

    //ds if we have multiple samples
    if (parameters->number_of_samples > 1) {

      //ds store precision recall values for later processing
      std::vector<std::pair<double, double>> precision_recall = parameters->evaluator->computePrecisionRecallCurve(closures, maximum_f1_score, 1);
      std::cerr << "maximum F1 score: " << maximum_f1_score << " [" << std::endl;
    } else {

      //ds directly write precision recall results to file
      parameters->evaluator->computePrecisionRecallCurve(closures, maximum_f1_score, 1, "precision_recall-"+benchmark_suffix);
    }
    maximum_f1_scores.push_back(maximum_f1_score);

    //ds add initial durations
    if (mean_processing_times.empty()) {
      mean_processing_times = matcher->durationsSecondsQueryAndTrain();
    } else {

      //ds add up all elements
      for (uint64_t u = 0; u < mean_processing_times.size(); ++u) {
        mean_processing_times[u] += matcher->durationsSecondsQueryAndTrain()[u];
      }
    }
  }

#ifdef SRRG_BENCH_BUILD_VIEWERS
  if (parameters->use_gui) {
    cv::destroyAllWindows();

    //ds allocate a qt UI server in the main scope (required)
    std::shared_ptr<QApplication> ui_server = std::make_shared<QApplication>(argc_, argv_);

    //ds allocate a viewer to check the resulting map qualitatively
    std::shared_ptr<ClosureViewer> viewer = std::make_shared<ClosureViewer>(parameters->method_name);

    //ds set closure map to display
    viewer->update(parameters->evaluator->imagePosesGroundTruth(),
                 parameters->evaluator->closureMap(),
                 parameters->evaluator->validClosures(),
                 parameters->evaluator->invalidClosures());

    //ds show the viewer
    viewer->show();

    //ds while the viewer is open
    while (viewer->isVisible()) {

    //ds update GL
    viewer->updateGL();
    ui_server->processEvents();

    //ds breathe
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
#endif

  std::cerr << std::endl;
  std::cerr << "maximum F1 scores: " << std::endl;
  for (const double& maximum_f1_score: maximum_f1_scores) {
    std::cerr << maximum_f1_score << std::endl;
  }

  //ds save f1 score to file (appending!)
  std::ofstream outfile_f1_score("maximum_f1_scores.txt", std::ifstream::app);
  outfile_f1_score << benchmark_suffix << " " << maximum_f1_scores.back() << std::endl;
  outfile_f1_score.close();

  //ds compute mean processing time over all samples
  double mean_processing_time = 0;
  for (uint64_t u = 0; u < mean_processing_times.size(); ++u) {
    mean_processing_times[u] /= parameters->number_of_samples;
    mean_processing_time += mean_processing_times[u];
  }
  mean_processing_time /= mean_processing_times.size();
  std::cerr << "mean processing time per image (s): " << mean_processing_time << std::endl;

  //ds compute mean, variance and standard deviation for query and train duration
  double variance_duration = 0;
  for (const double& duration_seconds: mean_processing_times) {
    variance_duration += (duration_seconds-mean_processing_time)*(duration_seconds-mean_processing_time);
  }
  variance_duration /= mean_processing_times.size();
  const double standard_deviation_duration = std::sqrt(variance_duration);

//  //ds compute mean, variance and standard deviation of match numbers returned (for correct closures)
//  double mean_number_of_matches       = 0;
//  std::vector<uint64_t> number_of_matches_per_closure(0);
//  for (const SearchResult& closure: closures) {
//    if (closure.image_association.valid) {
//      number_of_matches_per_closure.push_back(closure.number_of_matches_absolute);
//      mean_number_of_matches += closure.number_of_matches_absolute;
//    }
//  }
//  if (number_of_matches_per_closure.size() > 0) {
//
//    //ds dump first 1000 number of matches (or less)
//    std::ofstream outfile_matches("number-of-matches_"+benchmark_suffix, std::ifstream::out);
//    std::sort(number_of_matches_per_closure.begin(), number_of_matches_per_closure.end(), std::greater<uint64_t>());
//    for (uint64_t u = 0; u < (number_of_matches_per_closure.size() && 1000); ++u) {
//      outfile_matches << number_of_matches_per_closure[u] << std::endl;
//    }
//    outfile_matches.close();
//
//    //ds compute mean number of matches
//    mean_number_of_matches /= number_of_matches_per_closure.size();
//  }
//  double variance_number_of_matches = 0;
//  for (const SearchResult& closure: closures) {
//    if (closure.image_association.valid) {
//      variance_number_of_matches += (closure.number_of_matches_absolute-mean_number_of_matches)*(closure.number_of_matches_absolute-mean_number_of_matches);
//    }
//  }
//  if (number_of_matches_per_closure.size() > 0) {
//    variance_number_of_matches /= number_of_matches_per_closure.size();
//  }
//  const double standard_deviation_number_of_matches = std::sqrt(variance_number_of_matches);

  //ds save miscellaneous statistics to file (appending!)
  std::ofstream outfile_miscellaneous_statistics("statistics.txt", std::ifstream::app);
  outfile_miscellaneous_statistics << benchmark_suffix
                                   << " " << mean_processing_time << " " << standard_deviation_duration
                                   << " " << 0 << " " << 0
                                   << " " << 0 << " " << 0 <<  std::endl;
  outfile_miscellaneous_statistics.close();

  //ds save timing results to file with continuous image indices for plotting (of last run)
  std::ofstream outfile_duration("duration_"+ benchmark_suffix, std::ifstream::out);
  outfile_duration << "#0:IMAGE_NUMBER #1:MEAN_PROCESSING_TIME_SECONDS #2:STANDARD_DEVIATION" << std::endl;
  for (uint64_t u = 0; u < mean_processing_times.size(); ++u) {
    outfile_duration << u << " " << mean_processing_times[u] << " " << standard_deviation_duration << std::endl;
  }
  outfile_duration.close();

  //ds done
  std::cerr << "done" << std::endl;
  return EXIT_SUCCESS;
}
