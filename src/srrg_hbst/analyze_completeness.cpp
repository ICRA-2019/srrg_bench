#include "utilities/loop_closure_evaluator.h"
#include "srrg_hbst_types/binary_tree.hpp"

#if CV_MAJOR_VERSION == 2
  //ds no specifics
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

using namespace srrg_hbst;

//ds descriptor configuration
#define DESCRIPTOR_SIZE_BYTES 32
#define DESCRIPTOR_SIZE_BITS DESCRIPTOR_SIZE_BYTES*8
typedef BinaryMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef Matchable::Descriptor Descriptor;
typedef BinaryNode<Matchable> Node;
typedef Node::MatchableVector MatchableVector;
typedef BinaryTree<Node> Tree;
typedef Eigen::Matrix<double, DESCRIPTOR_SIZE_BITS, 1> EigenBitset;

//ds global image operators
std::shared_ptr<cv::BFMatcher> matcher;
std::shared_ptr<srrg_bench::LoopClosureEvaluator> evaluator;
cv::Ptr<cv::FeatureDetector> feature_detector;
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

//ds evaluated search structures
std::shared_ptr<Tree> hbst_balanced;
std::shared_ptr<Tree> hbst_incremental;

void loadMatchables(MatchableVector& matchables_,
                    const std::set<const srrg_bench::ImageWithPose*>& images_,
                    const uint64_t& target_number_of_descriptors_per_image_);

const uint64_t getNumberOfMatches(const Matchable* query_descriptor_,
                                  const MatchableVector& input_descriptors_,
                                  const uint32_t& maximum_distance_matching_);

const double getMeanRelativeNumberOfMatches(const std::shared_ptr<Tree> tree_,
                                            const MatchableVector& query_descriptors_,
                                            const std::map<const Matchable*, uint64_t>& feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_matching_);

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 5) {
    std::cerr << "invalid call - please use: ./descriptor_bit_analysis -images <KITTI_images_folder> -poses <KITTI_file_poses_ground_truth>"
                 "[-space <image_number_interspace> -descriptor <brief/orb/akaze/brisk -t <detector_threshold> -n <target_number_of_descriptors>]" << std::endl;
    return EXIT_FAILURE;
  }

  //ds configuration
  std::string images_folder                = "";
  std::string file_name_poses_ground_truth = "";
  std::string descriptor_type              = "brief";
  double detector_threshold                = 10;
  uint32_t target_number_of_descriptors_per_image = 1000;
  uint32_t target_depth                    = 17;
  uint32_t maximum_distance_matching       = 25;

  //ds test mode 0: resulting bit-wise completeness at depth 1 for a bit index k
  //ds test mode 1: resulting mean completeness for multiple depths, choosing the balanced k
  uint32_t test = 0;

  //ds scan the command line for configuration file input
  int32_t argc_parsed = 1;
  while(argc_parsed < argc_){
    if (!std::strcmp(argv_[argc_parsed], "-images")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      images_folder = argv_[argc_parsed];
    } else if(!std::strcmp(argv_[argc_parsed], "-poses")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      file_name_poses_ground_truth = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-descriptor")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      descriptor_type = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-t")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      detector_threshold = std::stod(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-n")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      target_number_of_descriptors_per_image = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-m")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      maximum_distance_matching = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-test")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      test = std::stoi(argv_[argc_parsed]);
    }
    argc_parsed++;
  }
  if (images_folder.empty()) {
    std::cerr << "ERROR: no images specified (use -images <images_folder>)" << std::endl;
    return EXIT_FAILURE;
  }
  if (file_name_poses_ground_truth.empty()) {
    std::cerr << "ERROR: no images specified (use -poses <file_poses_ground_truth>)" << std::endl;
    return EXIT_FAILURE;
  }

  //ds image name pattern
  const std::string image_name_pattern = "camera_left.image_raw_";
  const std::string image_file_format  = ".pgm";
  const std::string suffix             = "-"+descriptor_type+"-"+std::to_string(target_number_of_descriptors_per_image)+".txt";

  //ds log benchmark configuration
  std::cerr << "--------------------------------- CONFIGURATION ---------------------------------" << std::endl;
  LOG_VARIABLE(test)
  LOG_VARIABLE(images_folder)
  LOG_VARIABLE(image_name_pattern)
  LOG_VARIABLE(file_name_poses_ground_truth)
  LOG_VARIABLE(descriptor_type)
  LOG_VARIABLE(maximum_distance_matching)
  LOG_VARIABLE(detector_threshold)
  LOG_VARIABLE(target_number_of_descriptors_per_image)
  LOG_VARIABLE(suffix)
  LOG_VARIABLE(DESCRIPTOR_SIZE_BYTES)
  std::cerr << "---------------------------------------------------------------------------------" << std::endl;

  //ds enable multithreading
  cv::setNumThreads(4);

  //ds grab a loop closure evaluator instance
  evaluator = std::make_shared<srrg_bench::LoopClosureEvaluator>();

  //ds load ground truth poses
  evaluator->loadImagesWithPosesFromFileKITTI(file_name_poses_ground_truth, images_folder);

  //ds compute ground truth
  evaluator->computeLoopClosureFeasibilityMap(1, 1, M_PI/10, 500);
  LOG_VARIABLE(evaluator->totalNumberOfValidClosures())

  //ds feature handling
#if CV_MAJOR_VERSION == 2
  feature_detector = new cv::FastFeatureDetector(detector_threshold);
#elif CV_MAJOR_VERSION == 3
  feature_detector = cv::FastFeatureDetector::create(detector_threshold);
#endif

  //ds chose descriptor extractor
#if CV_MAJOR_VERSION == 2
  if (descriptor_type == "brief") {
    descriptor_extractor = new cv::BriefDescriptorExtractor(DESCRIPTOR_SIZE_BYTES);
  } else if (descriptor_type == "orb") {
    descriptor_extractor = new cv::ORB();
  } else if (descriptor_type == "brisk") {
    descriptor_extractor = new cv::BRISK();
  } else {
    std::cerr << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    return EXIT_FAILURE;
  }
#elif CV_MAJOR_VERSION == 3
  if (descriptor_type == "brief") {
    descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTOR_SIZE_BYTES); //DESCRIPTOR_SIZE_BITS bits
  } else if (descriptor_type == "orb") {
    feature_detector = cv::ORB::create(2*target_number_of_descriptors_per_image);
    descriptor_extractor = cv::ORB::create(); //256 bits
  } else if (descriptor_type == "brisk") {
    feature_detector = cv::BRISK::create(detector_threshold);
    descriptor_extractor = cv::BRISK::create(); //512 bits
  } else if (descriptor_type == "freak") {
    descriptor_extractor = cv::xfeatures2d::FREAK::create(); //512 bits
  } else if (descriptor_type == "akaze") {
    feature_detector     = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, detector_threshold); //486 bits
    descriptor_extractor = cv::AKAZE::create(); //486 bits
  } else {
    std::cerr << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    return EXIT_FAILURE;
  }
#endif

  //ds empty tree handles
  hbst_balanced    = std::make_shared<Tree>();
  hbst_incremental = std::make_shared<Tree>();

  //ds compute all sets of input descriptors
  MatchableVector input_descriptors_total;
  std::cerr << "computing input image descriptors: " << std::endl;
  loadMatchables(input_descriptors_total, evaluator->validTrainImagesWithPoses(), target_number_of_descriptors_per_image);
  std::cerr << std::endl;
  LOG_VARIABLE(input_descriptors_total.size());
  const uint64_t number_of_input_images(evaluator->validTrainImagesWithPoses().size());

  //ds compute all sets of query descriptors
  MatchableVector query_descriptors_total;
  std::cerr << "computing query image descriptors: " << std::endl;
  loadMatchables(query_descriptors_total, evaluator->validQueryImagesWithPoses(), target_number_of_descriptors_per_image);
  std::cerr << std::endl;
  LOG_VARIABLE(query_descriptors_total.size());

  //ds for all query descriptors
  std::cerr << "computing total number of feasible matches N_M in root node for tau: " << maximum_distance_matching
            << " complexity: " << static_cast<double>(query_descriptors_total.size())*input_descriptors_total.size() << std::endl;
  double mean_number_of_matches_per_query = 0;
  std::map<const Matchable*, uint64_t> feasible_number_of_matches_per_query;
  for (const Matchable* query_descriptor: query_descriptors_total) {
    feasible_number_of_matches_per_query.insert(std::make_pair(query_descriptor, 0));

    //ds against all input descriptors
    for (const Matchable* input_descriptor: input_descriptors_total) {

      //ds if the distance is within the threshold
      if ((query_descriptor->descriptor^input_descriptor->descriptor).count() < maximum_distance_matching) {
        ++feasible_number_of_matches_per_query.at(query_descriptor);
        ++mean_number_of_matches_per_query;
      }
    }
    std::cerr << "completed: " << feasible_number_of_matches_per_query.size() << "/" << query_descriptors_total.size() << std::endl;
  }
  mean_number_of_matches_per_query /= input_descriptors_total.size();
  LOG_VARIABLE(mean_number_of_matches_per_query);

  //ds depending on the mode
  if (test == 0) {

    //ds bitwise statistics
    EigenBitset mean_completeness(EigenBitset::Zero());
    EigenBitset bit_means(EigenBitset::Zero());

    //ds for each bit index
    for (uint32_t k = 0; k < DESCRIPTOR_SIZE_BITS; ++k) {
      double bit_values_accumulated = 0;

      //ds partition the input set according to the checked bit
      MatchableVector input_descriptors_left;
      MatchableVector input_descriptors_right;
      for (const Matchable* input_descriptor: input_descriptors_total) {
        if (input_descriptor->descriptor[k]) {
          input_descriptors_right.push_back(input_descriptor);
          ++bit_values_accumulated;
        } else {
          input_descriptors_left.push_back(input_descriptor);
        }
      }

      //ds compute bit mean for current index
      bit_means[k] = bit_values_accumulated/input_descriptors_total.size();

      //ds counting
      double relative_number_of_matches_accumulated = 0;

      //ds match all queries in the corresponding leafs to compute the completeness
      for (const Matchable* query_descriptor: query_descriptors_total) {
        uint64_t number_of_matches = 0;
        if (query_descriptor->descriptor[k]) {

          //ds match against all descriptors in right leaf
          number_of_matches = getNumberOfMatches(query_descriptor, input_descriptors_right, maximum_distance_matching);
        } else {

          //ds match against all descriptors in left leaf
          number_of_matches = getNumberOfMatches(query_descriptor, input_descriptors_left, maximum_distance_matching);
        }

        //ds update completeness for this descriptor and bit
        if (feasible_number_of_matches_per_query.at(query_descriptor) > 0) {
          relative_number_of_matches_accumulated += static_cast<double>(number_of_matches)/feasible_number_of_matches_per_query.at(query_descriptor);
        } else {
          relative_number_of_matches_accumulated += 1;
        }
      }

      //ds compute mean completeness over all queries for the current bit index
      mean_completeness[k] = relative_number_of_matches_accumulated/query_descriptors_total.size();
      std::cerr << "completed: " << k << "/" << DESCRIPTOR_SIZE_BITS << " : " << mean_completeness[k] << std::endl;
    }

    //ds save completeness to file
    std::ofstream outfile_bitwise_completeness("bitwise_completeness-"
                                               +std::to_string(evaluator->totalNumberOfValidClosures())+"-"
                                               +std::to_string(maximum_distance_matching)+suffix, std::ifstream::out);
    outfile_bitwise_completeness << "#0:BIT_INDEX 1:BIT_COMPLETENESS 2:BIT_MEAN" << std::endl;
    outfile_bitwise_completeness << std::fixed;
    for (uint32_t k = 0; k < DESCRIPTOR_SIZE_BITS; ++k) {
      outfile_bitwise_completeness << k << " " << mean_completeness[k] << " " << bit_means[k] << std::endl;
    }
    outfile_bitwise_completeness.close();

  } else {

    //ds initial estimate for comparison
    double completeness_first_split = 0;

    //ds save completeness to file
    std::ofstream outfile_cumulative_completeness("cumulative_completeness-"
                                                  +std::to_string(evaluator->totalNumberOfValidClosures())+"-"
                                                  +std::to_string(maximum_distance_matching)+suffix, std::ifstream::out);
    outfile_cumulative_completeness << "#0:DEPTH 1:PREDICTION 2:TOTAL 3:INCREMENTAL" << std::endl;
    outfile_cumulative_completeness << std::fixed;
    outfile_cumulative_completeness << "0 1.0 1.0 1.0" << std::endl;

    //ds initial situation
    std::printf("depth: %2i C(h) = P: %4.2f T: %4.2f I: %4.2f\n", 0, 1.0, 1.0, 1.0);

    //ds start trials for different depths
    for (uint32_t depth = 1; depth < target_depth; ++depth) {

      //ds construct tree with new maximum depth (only constraint)
      hbst_balanced->clear(false);
      hbst_incremental->clear(false);
      Node::maximum_partitioning = 0.5;
      Node::maximum_leaf_size    = 0;
      Node::maximum_depth        = depth;

      //ds construct total tree
      hbst_balanced->add(input_descriptors_total);

      //ds construct incremental tree - in batches
      for (uint64_t number_of_insertions = 0; number_of_insertions < number_of_input_images; ++number_of_insertions) {
        hbst_incremental->add(MatchableVector(input_descriptors_total.begin()+number_of_insertions*target_number_of_descriptors_per_image,
                                              input_descriptors_total.begin()+(number_of_insertions+1)*target_number_of_descriptors_per_image));
      }

      //ds obtain mean of relative number of matches
      const double mean_completeness_balanced    = getMeanRelativeNumberOfMatches(hbst_balanced,
                                                                                  query_descriptors_total,
                                                                                  feasible_number_of_matches_per_query,
                                                                                  maximum_distance_matching);
      const double mean_completeness_incremental = getMeanRelativeNumberOfMatches(hbst_incremental,
                                                                                  query_descriptors_total,
                                                                                  feasible_number_of_matches_per_query,
                                                                                  maximum_distance_matching);

      //ds buffer first split
      if (depth == 1) {
        completeness_first_split = mean_completeness_balanced;
      }
      const double mean_completeness_prediction = std::pow(completeness_first_split, depth);

      //ds print current completeness: I(ncremental) T(otal) E(stimated)
      std::printf("depth: %2u C(h) = P: %4.2f T: %4.2f I: %4.2f\n",
                  depth,
                  mean_completeness_prediction,
                  mean_completeness_balanced,
                  mean_completeness_incremental);
      outfile_cumulative_completeness << depth << " "
                                      << mean_completeness_prediction << " "
                                      << mean_completeness_balanced << " "
                                      << mean_completeness_incremental << std::endl;
    }
    outfile_cumulative_completeness.close();
  }

  //ds clear trees (without freeing matchables)
  hbst_balanced->clear(false);
  hbst_incremental->clear(false);

  //ds free all matchables
  for (const Matchable* matchable: query_descriptors_total) {
    delete matchable;
  }
  for (const Matchable* matchable: input_descriptors_total) {
    delete matchable;
  }

  //ds done
  return EXIT_SUCCESS;
}

void loadMatchables(MatchableVector& matchables_,
                    const std::set<const srrg_bench::ImageWithPose*>& images_,
                    const uint64_t& target_number_of_descriptors_per_image_) {
  matchables_.clear();
  for (const srrg_bench::ImageWithPose* image_with_pose: images_) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //ds load image
    cv::Mat image = cv::imread(image_with_pose->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints
    feature_detector->detect(image, keypoints);

    //ds sort keypoints descendingly by response value (for culling after descriptor computation)
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});

    //ds compute BRIEF descriptors for sorted keypoints
    descriptor_extractor->compute(image, keypoints, descriptors);

    //ds check insufficient descriptor number
    if (keypoints.size() < target_number_of_descriptors_per_image_) {
      std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints.size()
                << " < " << target_number_of_descriptors_per_image_ << ", adjust detector threshold\n" << std::endl;
      throw std::runtime_error("insufficient number of descriptors computed");
    }

    //ds compute matchables and store them
    MatchableVector matchables_current(hbst_balanced->getMatchablesWithIndex(descriptors(cv::Rect(0, 0, descriptors.cols, target_number_of_descriptors_per_image_)),
                                                                             image_with_pose->image_number));
    matchables_.insert(matchables_.end(), matchables_current.begin(), matchables_current.end());
    std::cerr << "x";
  }
}

const uint64_t getNumberOfMatches(const Matchable* query_descriptor_,
                                  const MatchableVector& input_descriptors_,
                                  const uint32_t& maximum_distance_matching_) {
  uint64_t number_of_matches = 0;
  for (const Matchable* input_descriptor: input_descriptors_) {
    if ((query_descriptor_->descriptor^input_descriptor->descriptor).count() < maximum_distance_matching_) {
      ++number_of_matches;
    }
  }
  return number_of_matches;
}

const double getMeanRelativeNumberOfMatches(const std::shared_ptr<Tree> tree_,
                                            const MatchableVector& query_descriptors_,
                                            const std::map<const Matchable*, uint64_t>& feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_matching_) {

  //ds relative number of matches summed up over all queries
  double relative_number_of_matches_accumulated = 0;

  //ds for each query descriptor
  for (const Matchable* query_descriptor: query_descriptors_) {

    //ds traverse the tree until no children are available -> we hit a leaf
    const Node* iterator = tree_->root();
    while (iterator->left) {
      if (query_descriptor->descriptor[iterator->index_split_bit]) {
        iterator = iterator->right;
      } else {
        iterator = iterator->left;
      }
    }

    //ds compute matches within threshold in this leaf
    uint64_t number_of_matches = getNumberOfMatches(query_descriptor, iterator->matchables, maximum_distance_matching_);

    //ds compute completeness
    if (feasible_number_of_matches_per_query_.at(query_descriptor) > 0) {
      relative_number_of_matches_accumulated += static_cast<double>(number_of_matches)/feasible_number_of_matches_per_query_.at(query_descriptor);
    } else {
      relative_number_of_matches_accumulated += 1;
    }
  }
  return relative_number_of_matches_accumulated/query_descriptors_.size();
}
