#include "../matchers/bruteforce_matcher.h"
#include "../utilities/command_line_parameters.h"

#ifdef SRRG_BENCH_BUILD_VIEWERS
#include <thread>
#include "qapplication.h"
#include "visualization/closure_viewer.h"
#endif

using namespace srrg_bench;

//ds feature map, required for geometric verification
std::map<ImageNumber, std::vector<cv::KeyPoint>> keypoints_per_image;

int32_t main(int32_t argc_, char** argv_) {
  std::cerr << std::fixed;

  //ds input validation
  if (argc_ < 4) {
    std::cerr << "ERROR: invalid call - use: /compute_closure_ground_truth [-cross] -mode <kitti/malaga/lucia/oxford> -images <folder_images_0> [<folder_images_1>] -poses <poses_gt_0> [<poses_gt_1>]"
                 "[-t <fast_threshold> -space <image_number_interspace> -p <difference_position_meters> -a <difference_angle_radians> -timestamps <lucia_timestamps> "
                 "-use-gui (-ug) -compute-confusion (-cc) -geometric-verification (-gv) <calibration_file> -recall (-r) -n <number_of_descriptors>]" << std::endl;
    return EXIT_FAILURE;
  }

  //ds default arguments
  bool compute_plausible_confusion    = false;
  bool compute_geometric_verification = false;

  //ds geometric verification - camera matrix parameters
  double maximum_essential_error    = 0.1;
  std::string file_name_calibration = "";
  double f_u = 0;
  double f_v = 0;
  double c_u = 0;
  double c_v = 0;

  //ds scan the command line for configuration file input
  int32_t argc_parsed = 1;
  while(argc_parsed < argc_){
    if (!std::strcmp(argv_[argc_parsed], "-compute-confusion") || !std::strcmp(argv_[argc_parsed], "-cc")) {
      compute_plausible_confusion = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-geometric-verification") || !std::strcmp(argv_[argc_parsed], "-gv")) {
      compute_geometric_verification = true;
      ++argc_parsed; if (argc_parsed == argc_) {break;}
      file_name_calibration = argv_[argc_parsed];
    }
    argc_parsed++;
  }
  keypoints_per_image.clear();

  //ds grab remaining configuration
  std::shared_ptr<CommandLineParameters> parameters = std::make_shared<CommandLineParameters>();
  parameters->parse(argc_, argv_);

  //ds relax parameters if geometric verification is performed
  if (compute_geometric_verification) {
    parameters->maximum_difference_angle_radians *= 2;
  }

  //ds setup
  parameters->validate(std::cerr);
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds if geometric verification is desired - confusion will always computed
  if (compute_geometric_verification) {
    compute_plausible_confusion = true;
  }
  LOG_VARIABLE(compute_plausible_confusion)
  LOG_VARIABLE(compute_geometric_verification)

  //ds if geometric verfication is desired
  if (compute_geometric_verification) {
    if (file_name_calibration == "" && parameters->parsing_mode != "lucia") {
      std::cerr << "ERROR: no camera calibration file provided for geometric verification" << std::endl;
      return EXIT_FAILURE;
    }
    LOG_VARIABLE(file_name_calibration)

    //ds depending on the mode TODO move to parameters handler
    if (parameters->parsing_mode == "kitti") {
      std::ifstream file_calibration(file_name_calibration, std::ifstream::in);
      std::string line_buffer("");
      std::getline(file_calibration, line_buffer);
      if (line_buffer.empty()) {
        std::cerr << "ERROR: invalid camera calibration file provided" << std::endl;
        return EXIT_FAILURE;
      }
      std::istringstream stream(line_buffer);

      //ds parse in fixed order
      std::string filler("");
      stream >> filler;
      stream >> f_u;
      stream >> filler;
      stream >> c_u;
      stream >> filler;
      stream >> filler;
      stream >> f_v;
      stream >> c_v;
      file_calibration.close();
    } else if (parameters->parsing_mode == "malaga") {

      //ds camera calibration for rectified images of malaga
      std::ifstream file_calibration(file_name_calibration, std::ifstream::in);
      std::string line_buffer("");
      std::getline(file_calibration, line_buffer);
      if (line_buffer.empty()) {
        std::cerr << "ERROR: invalid camera calibration file provided" << std::endl;
        return EXIT_FAILURE;
      }

      //ds keep reading - we're interested in line 9-12
      while (std::getline(file_calibration, line_buffer)) {

        //ds get indices for value parsing
        const std::string::size_type value_begin = line_buffer.find_first_of("=");
        const std::string::size_type value_end   = value_begin+9;

        //ds skip processing if no number is present in this format
        if (value_begin == std::string::npos || value_end > line_buffer.length()) {
          continue;
        }

        //ds check for parameters
        if (line_buffer.substr(0, 2) == "cx") {
          c_u = std::stod(line_buffer.substr(value_begin+1, value_end-value_begin));
        } else if (line_buffer.substr(0, 2) == "cy") {
          c_v = std::stod(line_buffer.substr(value_begin+1, value_end-value_begin));
        } else if (line_buffer.substr(0, 2) == "fx") {
          f_u = std::stod(line_buffer.substr(value_begin+1, value_end-value_begin));
        } else if (line_buffer.substr(0, 2) == "fy") {
          f_v = std::stod(line_buffer.substr(value_begin+1, value_end-value_begin));
          break;
        }
      }
      file_calibration.close();
    } else if (parameters->parsing_mode == "lucia") {

      //ds hardcoded for now (as there is only one sequence)
      f_u = 1246.56167;
      f_v = 1247.09234;
      c_u = 532.28794;
      c_v = 383.60407;
    } else {
      std::cerr << "ERROR: geometric verification is not implemented for the target parsing mode" << std::endl;
      return EXIT_FAILURE;
    }

    LOG_VARIABLE(f_u)
    LOG_VARIABLE(f_v)
    LOG_VARIABLE(c_u)
    LOG_VARIABLE(c_v)
    LOG_VARIABLE(maximum_essential_error)
  }

  //ds info
  double relative_number_of_points_removed_by_gv = 0;

  //ds check if full visibility matrix should be computed with brute-force
  if (compute_plausible_confusion) {

    //ds enable multithreading
    cv::setNumThreads(4);

    //ds allocate an exhaustive matcher
    std::shared_ptr<BruteforceMatcher> matcher = std::make_shared<BruteforceMatcher>(parameters->minimum_distance_between_closure_images,
                                                                                     parameters->distance_norm);

    //ds first load all images and compute the descriptors for each of them
    const uint32_t number_of_images = parameters->evaluator->numberOfImages();

    //ds 2d score matrix (queries to references)
    double** confusion_matrix = 0;
    confusion_matrix = new double*[number_of_images];
    for (uint32_t row = 0; row < number_of_images; ++row) {
      confusion_matrix[row] = new double[number_of_images];
      for (uint32_t col = 0; col < number_of_images; ++col) {
        confusion_matrix[row][col] = 0;
      }
    }

    //ds bookkeep maximum scores and associated query to reference matching
    std::vector<ResultDescriptorMatching> closures(0);
    ImageNumberQuery image_number_query = parameters->image_number_start;

    //ds for video processing
    if (parameters->parsing_mode == "nordland") {
      image_number_query = 0;

      //ds skip images up to start number
      while (parameters->video_player_query.grab() && image_number_query < parameters->image_number_start) {
        ++image_number_query;
        std::cerr << "skipped images: " << image_number_query << "\r";
      }
    }

    //ds for all images we have
    std::cerr << "computing plausible false positives using brute-force matching: " << std::endl;
    const uint32_t length_progress_bar   = 20;
    uint64_t number_of_descriptors_total = 0;
    std::vector<uint64_t> number_of_descriptors_accumulated(0);
    uint64_t number_of_skipped_images    = 0;
    while (image_number_query < parameters->image_number_stop) {

      //ds for video processing
      if (parameters->parsing_mode == "nordland") {

        //ds grab the next image and check for failure
        if (!parameters->video_player_query.grab()) {

          //ds escape
          std::cerr << "\nERROR: no more images in video stream" << std::endl;
          break;
        }
      }

      //ds if we got a query image
      if (image_number_query%parameters->query_interspace == 0) {

        //ds query image
        cv::Mat image;

        //ds for video processing
        if (parameters->parsing_mode == "nordland") {

          //ds decode image from stream
          parameters->video_player_query.retrieve(image);
          cv::cvtColor(image, image, CV_RGB2GRAY);
        } else {

          //ds load image from disk
          image = cv::imread(parameters->evaluator->imagePosesGroundTruth()[image_number_query]->file_path, CV_LOAD_IMAGE_GRAYSCALE);
        }

        //ds apply bayer decoding if necessary
        if (parameters->parsing_mode == "lucia" || parameters->parsing_mode == "oxford") {
          cv::cvtColor(image, image, CV_BayerGR2GRAY);
        }

        //ds detect keypoints
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
          ++image_number_query;
          ++number_of_skipped_images;
          continue;
        }

        //ds rebuild descriptor matrix and keypoints vector
        keypoints.resize(parameters->target_number_of_descriptors);
        descriptors = descriptors(cv::Rect(0, 0, descriptors.cols, parameters->target_number_of_descriptors));

        //ds backup keypoints for geometric verification
        keypoints_per_image.insert(std::make_pair(image_number_query, keypoints));

        //ds update statistics
        number_of_descriptors_total += descriptors.rows;
        number_of_descriptors_accumulated.push_back(number_of_descriptors_total);

        //ds query against all past images, retrieving closures with relative scores
        std::vector<ResultDescriptorMatching> closures_current_query(0);
        matcher->query(descriptors, image_number_query, parameters->maximum_descriptor_distance, closures_current_query);

        //ds update confusion matrix for the current query
        for (ResultDescriptorMatching& result: closures_current_query) {
          if (compute_geometric_verification) {
          uint64_t number_of_valid_associations = 0;

          //ds perform geometric verification for all points - compute essential matrix
          const Eigen::Isometry3d& query_to_world(parameters->evaluator->imagePosesGroundTruth().at(result.result_image_retrieval.image_association.query)->pose);
          const Eigen::Isometry3d& train_to_world(parameters->evaluator->imagePosesGroundTruth().at(result.result_image_retrieval.image_association.train)->pose);
          const Eigen::Isometry3d query_to_train = train_to_world.inverse()*query_to_world;
          Eigen::Matrix3d translation_skew;
          translation_skew << 0, -query_to_train.translation().z(), query_to_train.translation().y(),
                              query_to_train.translation().z(), 0, -query_to_train.translation().x(),
                              -query_to_train.translation().y(), query_to_train.translation().x(), 0;
          const Eigen::Matrix3d essential_matrix = query_to_train.linear()*translation_skew;

          //ds for every keypoint correspondence
          for (const IndexAssociation& descriptor_association: result.descriptor_associations) {

            //ds obtain normalized point coordinates
            const double u_query = (keypoints_per_image.at(result.result_image_retrieval.image_association.query)[descriptor_association.index_query].pt.x-c_u)/f_u;
            const double v_query = (keypoints_per_image.at(result.result_image_retrieval.image_association.query)[descriptor_association.index_query].pt.y-c_v)/f_v;
            const double u_train = (keypoints_per_image.at(result.result_image_retrieval.image_association.train)[descriptor_association.index_train].pt.x-c_u)/f_u;
            const double v_train = (keypoints_per_image.at(result.result_image_retrieval.image_association.train)[descriptor_association.index_train].pt.y-c_v)/f_v;
            const Eigen::Vector3d image_coordinates_query(u_query, v_query, 1);
            const Eigen::Vector3d image_coordinates_train(u_train, v_train, 1);

            //ds compute essential error
            const double error = std::fabs(image_coordinates_train.transpose()*essential_matrix*image_coordinates_query);

            //ds check if association is consistent
            if (error < maximum_essential_error) {
              ++number_of_valid_associations;
            }
          }

          //ds update search result
          result.number_of_matches_relative_verified = static_cast<double>(number_of_valid_associations)/descriptors.rows;
          relative_number_of_points_removed_by_gv   += result.result_image_retrieval.number_of_matches_relative-result.number_of_matches_relative_verified;

            //ds update confusion matrix
            confusion_matrix[result.result_image_retrieval.image_association.query][result.result_image_retrieval.image_association.train]
            = result.number_of_matches_relative_verified;
          } else {

            //ds do not refine search result
            result.number_of_matches_relative_verified = result.result_image_retrieval.number_of_matches_relative;

            //ds update confusion matrix
            confusion_matrix[result.result_image_retrieval.image_association.query][result.result_image_retrieval.image_association.train]
            = result.result_image_retrieval.number_of_matches_relative;
          }
        }

        //ds train on the query image - so we can retrieve subsequent closures
        matcher->train(descriptors, image_number_query, keypoints);

        //ds accumulate closures over all queries
        closures.insert(closures.end(), closures_current_query.begin(), closures_current_query.end());

        //ds if desired - draw keypoints with descriptors
        if (parameters->use_gui) {
          cv::Mat image_display;
          cv::cvtColor(image, image_display, CV_GRAY2RGB);
          for (const cv::KeyPoint& keypoint: keypoints) {
            cv::circle(image_display, keypoint.pt, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(image_display, keypoint.pt, keypoint.size, cv::Scalar(0, 0, 255), 1);
          }

          //ds display currently detected keypoints
          cv::imshow("current image", image_display);
          cv::waitKey(1);
        }
      }

      //ds progress feedback: compute printing configuration
      const double progress           = static_cast<double>(matcher->numberOfQueries()+number_of_skipped_images)/parameters->number_of_images_to_process;
      const uint32_t length_completed = progress*length_progress_bar;

      //ds draw completed bar
      std::cerr << "progress [";
      for (uint32_t u = 0; u < length_completed; ++u) {
        std::cerr << "+";
      }
      for (uint32_t u = length_completed; u < length_progress_bar; ++u) {
        std::cerr << " ";
      }
      std::cerr << "] " << static_cast<int32_t>(progress*100.0) << " %"
                        << " | current image number: " << image_number_query
                        << " | TOTAL queries: " << matcher->numberOfQueries()
                        << " descriptors: " << number_of_descriptors_total
                        << " (average: " << static_cast<double>(number_of_descriptors_total)/matcher->numberOfQueries() << ")"
                        << " | current processing time (s): " << matcher->durationsSecondsQueryAndTrain().back() << "\r";
      ++image_number_query;
    }
    std::cerr << std::endl;
    if (parameters->use_gui) {
      cv::destroyAllWindows();
    }

    //ds compute average removal rate (for all image matches)
    relative_number_of_points_removed_by_gv /= closures.size();
    std::cerr << "average removed points by geometric verification: " << relative_number_of_points_removed_by_gv << std::endl;

    //ds mirror symmetric scores to obtain the symmetric confusion matrix
    for (uint32_t image_number_train = 0; image_number_train < number_of_images-parameters->minimum_distance_between_closure_images; ++image_number_train) {
      for (uint32_t image_number_query = image_number_train+parameters->minimum_distance_between_closure_images; image_number_query < number_of_images; ++image_number_query) {
        confusion_matrix[image_number_train][image_number_query] = confusion_matrix[image_number_query][image_number_train];
      }
    }

    //ds if we have geometrically verified data
    if (compute_geometric_verification) {

      //ds sort top scores in descending order - by geometrically verified matches
      std::sort(closures.begin(), closures.end(), [](const ResultDescriptorMatching& a_, const ResultDescriptorMatching& b_)
                {return a_.number_of_matches_relative_verified > b_.number_of_matches_relative_verified;});
    } else {

      //ds sort top scores in descending order
      std::sort(closures.begin(), closures.end(), [](const ResultDescriptorMatching& a_, const ResultDescriptorMatching& b_)
                {return a_.number_of_matches_relative_verified > b_.result_image_retrieval.number_of_matches_relative;});
    }

    //ds file suffix
    const std::string suffix = std::to_string(parameters->query_interspace) + "-"
                             + std::to_string(parameters->minimum_distance_between_closure_images) + "-"
                             + std::to_string(static_cast<int32_t>(parameters->maximum_descriptor_distance)) + "_"
                             + parameters->descriptor_type + "-"
                             + std::to_string(DESCRIPTOR_SIZE_BITS) + ".txt";

    //ds buffers
    ClosureMap closure_map_bf;
    const ClosureMap& closure_map_trajectory(parameters->evaluator->closureFeasibilityMap());
    ImageNumber total_number_of_valid_closures = 0;

    //ds compute plausible false positives appearing before reaching 100% recall
    for (const ResultDescriptorMatching& result: closures) {

      //ds compute current recall rate
      const double recall = static_cast<double>(total_number_of_valid_closures)/parameters->evaluator->totalNumberOfValidClosures();

      //ds if the query is contained in the ground truth and the reference is matching
      if (closure_map_trajectory.find(result.result_image_retrieval.image_association.query) != closure_map_trajectory.end() &&
          closure_map_trajectory.at(result.result_image_retrieval.image_association.query).count(result.result_image_retrieval.image_association.train)) {

        //ds if we haven't closed this query yet
        if (closure_map_bf.find(result.result_image_retrieval.image_association.query) == closure_map_bf.end()) {

          //ds enable the closure for ground truth
          std::multiset<ImageNumberTrain> train_image_numbers;
          train_image_numbers.insert(result.result_image_retrieval.image_association.train);
          closure_map_bf.insert(std::make_pair(result.result_image_retrieval.image_association.query, train_image_numbers));
        } else {

          //ds add the reference
          closure_map_bf.at(result.result_image_retrieval.image_association.query).insert(result.result_image_retrieval.image_association.train);
        }
      } else {

        //ds terminate as soon as a false positive would be reported by BF
        std::cerr << BAR << std::endl;
        std::cerr << "obtained false closure: " << result.result_image_retrieval.image_association.query
                                       << " > " << result.result_image_retrieval.image_association.train << std::endl;
        std::cerr << "query corresponding to: " << parameters->evaluator->imagePosesGroundTruth()[result.result_image_retrieval.image_association.query]->file_path
                  << "\nwith pose\n: " << parameters->evaluator->imagePosesGroundTruth()[result.result_image_retrieval.image_association.query]->pose.matrix() << std::endl;
        std::cerr << "reference corresponding to: " << parameters->evaluator->imagePosesGroundTruth()[result.result_image_retrieval.image_association.train]->file_path
                  << "\nwith pose\n: " << parameters->evaluator->imagePosesGroundTruth()[result.result_image_retrieval.image_association.train]->pose.matrix() << std::endl;
        std::cerr << "terminated (recall: " << recall << " closures: " << total_number_of_valid_closures << ")" << std::endl;
        std::cerr << BAR << std::endl;
        break;
      }

      //ds always added at this point
      ++total_number_of_valid_closures;
    }

    //ds dump closures to file: #QUERY #REFERENCE
    const std::string outfile_closures_ground_truth_name = "bf_closures_"+ suffix;
    std::ofstream outfile_closures_ground_truth(outfile_closures_ground_truth_name);
    outfile_closures_ground_truth << "#QUERY_IMAGE_IDENTIFIER #TRAIN_IMAGE_IDENTIFIER" << std::endl;
    for (const ClosureMapElement& element: closure_map_bf) {

      //ds dump each element in the set
      for (const ImageNumberTrain& image_number_reference: element.second) {
        outfile_closures_ground_truth << element.first << " " << image_number_reference << std::endl;
      }
    }
    outfile_closures_ground_truth.close();
    std::cerr << "closures saved to: " << outfile_closures_ground_truth_name
              << " total: " << total_number_of_valid_closures << std::endl;

    //ds dump score matrix to file
    const std::string outfile_confusion_matrix_name = "bf_confusion_matrix_"+ suffix;
    std::ofstream outfile_confusion_matrix(outfile_confusion_matrix_name);
    outfile_confusion_matrix.precision(3);
    outfile_confusion_matrix << std::fixed;
    for (uint32_t row = 0; row < number_of_images; ++row) {
      for (uint32_t col = 0; col < number_of_images; ++col) {
        outfile_confusion_matrix << confusion_matrix[row][col] << " ";
      }
      outfile_confusion_matrix << std::endl;
    }
    outfile_confusion_matrix.close();
    std::cerr << "Brute-force confusion matrix saved to: " << outfile_confusion_matrix_name << std::endl;

    //ds free score matrix
    for (ImageNumberQuery row = 0; row < number_of_images; ++row) {
      delete[] confusion_matrix[row];
    }
    delete[] confusion_matrix;
  }

#ifdef SRRG_BENCH_BUILD_VIEWERS

  //ds if GUI usage is defined
  if (parameters->use_gui) {

    //ds allocate a qt UI server in the main scope (required)
    std::shared_ptr<QApplication> ui_server = std::make_shared<QApplication>(argc_, argv_);

    //ds allocate a viewer to check the resulting map qualitatively
    std::shared_ptr<ClosureViewer> viewer = std::make_shared<ClosureViewer>();

    //ds set closure map to display
    viewer->update(parameters->evaluator->imagePosesGroundTruth(),
                   parameters->evaluator->closureFeasibilityMap(),
                   parameters->evaluator->validQueryImagesWithPoses(),
                   parameters->evaluator->validTrainImagesWithPoses(),
                   parameters->query_interspace);

    //ds show the viewer
    viewer->show();

    //ds adjust viewer dimensions depending on dataset
    if (parameters->parsing_mode == "nordland") {
      viewer->standardCamera()->setZNear(1);
      viewer->standardCamera()->setZFar(1e7);
    }

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

  return EXIT_SUCCESS;
}
