#include "../utilities/command_line_parameters.h"
#include "srrg_hbst_types/binary_tree.hpp"



//ds descriptor configuration
typedef srrg_hbst::BinaryMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef Matchable::Descriptor Descriptor;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef Node::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree<Node> Tree;
typedef Eigen::Matrix<double, DESCRIPTOR_SIZE_BITS, 1> EigenBitset;



//ds global image operators
std::shared_ptr<srrg_bench::LoopClosureEvaluator> evaluator;
cv::Ptr<cv::FeatureDetector> feature_detector;
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

//ds result containers
std::map<const Matchable*, uint64_t> feasible_number_of_matches_per_query;
std::vector<EigenBitset> accumulated_bitwise_completeness_per_depth;
std::vector<uint32_t> number_of_evaluated_leafs_per_depth; //ds the elements should have values of the sequence 1, 2, 4, 8, 16, 32, .. for increasing depths

void loadMatchables(MatchableVector& matchables_,
                    const std::set<const srrg_bench::ImageWithPose*>& images_,
                    const uint64_t& target_number_of_descriptors_per_image_);

const uint64_t getNumberOfMatches(const Matchable* query_descriptor_,
                                  const MatchableVector& input_descriptors_,
                                  const uint32_t& maximum_distance_matching_);

const MatchableVector getAvailableDescriptors(const MatchableVector& descriptors_,
                                              const std::vector<uint32_t>& splitting_bits_left_,
                                              const std::vector<uint32_t>& splitting_bits_right_);

void evaluateBitWiseCompleteness(const MatchableVector& parent_query_descriptors_,
                                 const MatchableVector& parent_reference_descriptors_,
                                 std::vector<uint32_t> splitting_bits_left_,
                                 std::vector<uint32_t> splitting_bits_right_,
                                 const uint32_t& maximum_depth_,
                                 const uint32_t& maximum_distance_hamming_);

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 5) {
    std::cerr << "invalid call - please use: ./analyze_completeness_leafs -images <KITTI_images_folder> -poses <KITTI_file_poses_ground_truth>"
                 "[-space <image_number_interspace> -descriptor <brief/orb/akaze/brisk -t <detector_threshold> -n <target_number_of_descriptors>]" << std::endl;
    return EXIT_FAILURE;
  }

  //ds grab configuration
  std::shared_ptr<srrg_bench::CommandLineParameters> parameters = std::make_shared<srrg_bench::CommandLineParameters>();
  parameters->parse(argc_, argv_);

  //ds configure
  parameters->validate(std::cerr);
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds set globals
  evaluator            = parameters->evaluator;
  feature_detector     = parameters->feature_detector;
  descriptor_extractor = parameters->descriptor_extractor;

  //ds compute all sets of input descriptors
  MatchableVector input_descriptors_total;
  std::cerr << "computing input image descriptors: " << std::endl;
  loadMatchables(input_descriptors_total, evaluator->validTrainImagesWithPoses(), parameters->target_number_of_descriptors);
  std::cerr << std::endl;
  LOG_VARIABLE(input_descriptors_total.size());

  //ds compute all sets of query descriptors
  MatchableVector query_descriptors_total;
  std::cerr << "computing query image descriptors: " << std::endl;
  loadMatchables(query_descriptors_total, evaluator->validQueryImagesWithPoses(), parameters->target_number_of_descriptors);
  std::cerr << std::endl;
  LOG_VARIABLE(query_descriptors_total.size());

  //ds for all query descriptors
  std::cerr << "computing total number of feasible matches (C_0=1) N_M in root node for tau: " << parameters->maximum_descriptor_distance
            << " complexity: " << static_cast<double>(query_descriptors_total.size())*input_descriptors_total.size() << std::endl;
  double mean_number_of_matches_per_query = 0;
  feasible_number_of_matches_per_query.clear();
  for (const Matchable* query_descriptor: query_descriptors_total) {
    feasible_number_of_matches_per_query.insert(std::make_pair(query_descriptor, 0));

    //ds against all input descriptors
    for (const Matchable* input_descriptor: input_descriptors_total) {

      //ds if the distance is within the threshold
      if ((query_descriptor->descriptor^input_descriptor->descriptor).count() < parameters->maximum_descriptor_distance) {
        ++feasible_number_of_matches_per_query.at(query_descriptor);
        ++mean_number_of_matches_per_query;
      }
    }
    if (feasible_number_of_matches_per_query.size()%1000 == 0) {
      std::cerr << "completed: " << feasible_number_of_matches_per_query.size() << "/" << query_descriptors_total.size() << std::endl;
    }
  }
  mean_number_of_matches_per_query /= input_descriptors_total.size();
  LOG_VARIABLE(mean_number_of_matches_per_query);

  //ds result containers
  accumulated_bitwise_completeness_per_depth.clear();
  number_of_evaluated_leafs_per_depth.clear();

  //ds chosen splitting bits (depth analysis) - propagated individually for each path
  std::vector<uint32_t> splitting_bits_left(0);
  std::vector<uint32_t> splitting_bits_right(0);

  //ds start recursive evaluation of descriptor splits - this will start recursive evaluations on left and right subtrees
  evaluateBitWiseCompleteness(query_descriptors_total,
                              input_descriptors_total,
                              splitting_bits_left,
                              splitting_bits_right,
                              parameters->maximum_depth,
                              parameters->maximum_descriptor_distance);

  //ds print results
  for (uint32_t d = 0; d < parameters->maximum_depth; ++d) {

    //ds compute mean bitwise vector at each depth (i.e. at depth=1 we have 2^1, at depth=2 we have 2^2 measurements)
    const EigenBitset mean_bitwise_completeness = accumulated_bitwise_completeness_per_depth[d]/number_of_evaluated_leafs_per_depth[d];

    //ds save completeness to file
    std::ofstream outfile_bitwise_completeness("bitwise_completeness-"
                                               +std::to_string(evaluator->totalNumberOfValidClosures())+"-"
                                               +std::to_string(parameters->maximum_descriptor_distance)+"_"
                                               +parameters->descriptor_type+"-"+std::to_string(DESCRIPTOR_SIZE_BITS)+"_depth-"
                                               +std::to_string(d)+".txt", std::ifstream::out);
    outfile_bitwise_completeness << "#0:BIT_INDEX 1:BIT_COMPLETENESS 2:BIT_MEAN" << std::endl;
    outfile_bitwise_completeness << std::fixed;
    for (uint32_t k = 0; k < DESCRIPTOR_SIZE_BITS; ++k) {
      outfile_bitwise_completeness << k << " " << mean_bitwise_completeness[k] << std::endl;
    }
    outfile_bitwise_completeness.close();
  }

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
    MatchableVector matchables_current(Tree::getMatchablesWithIndex(descriptors(cv::Rect(0, 0, descriptors.cols, target_number_of_descriptors_per_image_)),
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

const MatchableVector getAvailableDescriptors(const MatchableVector& descriptors_,
                                              const std::vector<uint32_t>& splitting_bits_left_,
                                              const std::vector<uint32_t>& splitting_bits_right_) {
  MatchableVector descriptors_available(0);
  for (const Matchable* descriptor: descriptors_) {

    //ds check if some of the splitting bits are set for the corresponding side (left or right)
    bool available = true;

    //ds check left splitting bits: descriptor is discarded if it matches a left split (would go right)
    for (const uint32_t& splitting_bit: splitting_bits_left_) {
      if (descriptor->descriptor[splitting_bit]) {
        available = false;
        break;
      }
    }

    //ds check right splitting bits: descriptor is discarded if it MISmatches a right split (would go left)
    for (const uint32_t& splitting_bit: splitting_bits_right_) {
      if (!descriptor->descriptor[splitting_bit]) {
        available = false;
        break;
      }
    }

    //ds skip the evaluation of this descriptor
    if (available) {
      descriptors_available.push_back(descriptor);
    }
  }
  return descriptors_available;
}

void evaluateBitWiseCompleteness(const MatchableVector& parent_query_descriptors_,
                                 const MatchableVector& parent_reference_descriptors_,
                                 std::vector<uint32_t> splitting_bits_left_,
                                 std::vector<uint32_t> splitting_bits_right_,
                                 const uint32_t& maximum_depth_,
                                 const uint32_t& maximum_distance_hamming_) {

  //ds splitting bit choice
  double best_bit_mean_distance = 0.5;
  uint32_t best_bit_index       = 0;

  //ds compute available descriptors in this leaf
  const MatchableVector query_descriptors(getAvailableDescriptors(parent_query_descriptors_, splitting_bits_left_, splitting_bits_right_));
  const MatchableVector reference_descriptors(getAvailableDescriptors(parent_reference_descriptors_, splitting_bits_left_, splitting_bits_right_));

  //ds current depth
  const uint32_t depth = splitting_bits_left_.size()+splitting_bits_right_.size();

  //ds check if we have to terminate because of missing descriptors
  if (query_descriptors.size() == 0) {
    std::cerr << "WARNING: no query descriptors at depth: " << depth
              << " parent descriptors: " << parent_query_descriptors_.size() << std::endl;
    return;
  }
  if (reference_descriptors.size() == 0) {
    std::cerr << "WARNING: no reference descriptors at depth: " << depth
              << " parent descriptors: " << parent_reference_descriptors_.size() << std::endl;
    return;
  }

  //ds bitwise statistics
  EigenBitset mean_completeness(EigenBitset::Zero());
  EigenBitset bit_means(EigenBitset::Zero());

  //ds check if we're the first leaf the current depth to evaluate - create result container
  if (number_of_evaluated_leafs_per_depth.size() == depth) {
    accumulated_bitwise_completeness_per_depth.push_back(mean_completeness);
    number_of_evaluated_leafs_per_depth.push_back(0);
  }

  //ds for each bit index
  for (uint32_t k = 0; k < DESCRIPTOR_SIZE_BITS; ++k) {
    double bit_values_accumulated = 0;

    //ds partition the input set according to the checked bit
    MatchableVector input_descriptors_left;
    MatchableVector input_descriptors_right;
    for (const Matchable* input_descriptor: reference_descriptors) {
      if (input_descriptor->descriptor[k]) {
        input_descriptors_right.push_back(input_descriptor);
        ++bit_values_accumulated;
      } else {
        input_descriptors_left.push_back(input_descriptor);
      }
    }

    //ds compute bit mean for current index
    bit_means[k] = bit_values_accumulated/reference_descriptors.size();

    //ds check if better and not already contained in a past splitting
    if (std::fabs(0.5-bit_means[k]) < best_bit_mean_distance &&
        std::find(splitting_bits_left_.begin(), splitting_bits_left_.end(), k) == splitting_bits_left_.end() &&
        std::find(splitting_bits_right_.begin(), splitting_bits_right_.end(), k) == splitting_bits_right_.end() ) {
      best_bit_mean_distance = std::fabs(0.5-bit_means[k]);
      best_bit_index = k;
    }

    //ds counting
    double relative_number_of_matches_accumulated = 0;

    //ds match all queries in the corresponding leafs to compute the completeness
    for (const Matchable* query_descriptor: query_descriptors) {
      uint64_t number_of_matches = 0;
      if (query_descriptor->descriptor[k]) {

        //ds match against all descriptors in right leaf
        number_of_matches = getNumberOfMatches(query_descriptor, input_descriptors_right, maximum_distance_hamming_);
      } else {

        //ds match against all descriptors in left leaf
        number_of_matches = getNumberOfMatches(query_descriptor, input_descriptors_left, maximum_distance_hamming_);
      }

      //ds update completeness for this descriptor and bit
      if (feasible_number_of_matches_per_query.at(query_descriptor) > 0) {
        relative_number_of_matches_accumulated += static_cast<double>(number_of_matches)/feasible_number_of_matches_per_query.at(query_descriptor);
      } else {
        relative_number_of_matches_accumulated += 1;
      }
    }

    //ds compute mean completeness over all queries for the current bit index
    mean_completeness[k] = relative_number_of_matches_accumulated/query_descriptors.size();
    std::cerr << "completed depth: " << depth << "/" << number_of_evaluated_leafs_per_depth[depth]
              << " bit index: " << k << "/" << DESCRIPTOR_SIZE_BITS << " : " << mean_completeness[k] << std::endl;
  }

  //ds update result containers
  accumulated_bitwise_completeness_per_depth[depth] += mean_completeness;
  ++number_of_evaluated_leafs_per_depth[depth];

  //ds if maximum depth is not yet reached
  if (depth < maximum_depth_) {

    //ds continue evaluating the potential splits on LEFT of the current node
    std::vector<uint32_t> splitting_bits_left_next(splitting_bits_left_);
    splitting_bits_left_next.push_back(best_bit_index);
    evaluateBitWiseCompleteness(query_descriptors,
                                reference_descriptors,
                                splitting_bits_left_next,
                                splitting_bits_right_,
                                maximum_depth_,
                                maximum_distance_hamming_);

    //ds continue evaluating the potential splits on RIGHT of the current node
    std::vector<uint32_t> splitting_bits_right_next(splitting_bits_right_);
    splitting_bits_right_next.push_back(best_bit_index);
    evaluateBitWiseCompleteness(query_descriptors,
                                reference_descriptors,
                                splitting_bits_left_,
                                splitting_bits_right_next,
                                maximum_depth_,
                                maximum_distance_hamming_);
  }
}
