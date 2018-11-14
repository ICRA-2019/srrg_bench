#include "../utilities/command_line_parameters.h"
#include "srrg_hbst/types/binary_tree.hpp"



//ds HBST configuration
typedef srrg_hbst::BinaryMatchable<uint64_t, DESCRIPTOR_SIZE_BITS> Matchable;
typedef Matchable::Descriptor Descriptor;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef Node::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree<Node> Tree;

//ds global image operators
std::shared_ptr<srrg_bench::LoopClosureEvaluator> evaluator;
cv::Ptr<cv::FeatureDetector> feature_detector;
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;
const uint32_t number_of_checked_bits_for_mean_guess = 32;



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

const double getMeanRelativeNumberOfMatches(const MatchableVector& query_descriptors_,
                                            const MatchableVector& reference_descriptors_,
                                            const std::map<const Matchable*, uint64_t>& feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_hamming_);

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 5) {
    std::cerr << "invalid call - please use: ./analyze_completeness_hbst "
                 "-mode kitti"
                 "-images <images_folder> "
                 "-poses <poses_ground_truth> "
                 "-depth <maximum_depth> "
                 "[...]" << std::endl;
    return EXIT_FAILURE;
  }

  //ds grab configuration
  std::shared_ptr<srrg_bench::CommandLineParameters> parameters = std::make_shared<srrg_bench::CommandLineParameters>();
  parameters->parse(argc_, argv_);

  //ds enforce hbst evaluation
  parameters->method_name = "hbst";

  //ds configure
  parameters->validate(std::cerr);
  parameters->configure(std::cerr);
  parameters->write(std::cerr);

  //ds set globals
  evaluator            = parameters->evaluator;
  feature_detector     = parameters->feature_detector;
  descriptor_extractor = parameters->descriptor_extractor;

  //ds empty tree handles
  std::shared_ptr<Tree> hbst_balanced    = std::make_shared<Tree>();
  std::shared_ptr<Tree> hbst_incremental = std::make_shared<Tree>();

  //ds compute all sets of input descriptors
  MatchableVector input_descriptors_total;
  std::cerr << "computing input image descriptors: " << std::endl;
  loadMatchables(input_descriptors_total, evaluator->validTrainImagesWithPoses(), parameters->target_number_of_descriptors);
  std::cerr << std::endl;
  LOG_VARIABLE(input_descriptors_total.size());
  const uint64_t number_of_input_images(evaluator->validTrainImagesWithPoses().size());

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
  std::map<const Matchable*, uint64_t> feasible_number_of_matches_per_query;
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

  //ds initial estimate for comparison
  double completeness_first_split = 0;

  //ds save completeness to file
  std::ofstream outfile_cumulative_completeness("cumulative_completeness-"
                                                +std::to_string(static_cast<uint32_t>(parameters->maximum_descriptor_distance))+"-"
                                                +parameters->descriptor_type+"-"+std::to_string(DESCRIPTOR_SIZE_BITS)+".txt", std::ifstream::out);
  outfile_cumulative_completeness << "#0:DEPTH 1:PREDICTION 2:TOTAL 3:INCREMENTAL" << std::endl;
  outfile_cumulative_completeness << std::fixed;
  outfile_cumulative_completeness << "0 1.0 1.0 1.0" << std::endl;

  //ds initial situation
  std::printf("depth: %2i C(h) = P: %4.2f T: %4.2f I: %4.2f\n", 0, 1.0, 1.0, 1.0);

  //ds start trials for different depths
  for (uint32_t depth = 1; depth < parameters->maximum_depth; ++depth) {

    //ds construct tree with new maximum depth (only constraint)
    hbst_balanced->clear(false);
    hbst_incremental->clear(false);
    Node::maximum_depth = depth;

    //ds construct total tree
    hbst_balanced->add(input_descriptors_total);

    //ds construct incremental tree - in batches
    for (uint64_t number_of_insertions = 0; number_of_insertions < number_of_input_images; ++number_of_insertions) {
      hbst_incremental->add(MatchableVector(input_descriptors_total.begin()+number_of_insertions*parameters->target_number_of_descriptors,
                                            input_descriptors_total.begin()+(number_of_insertions+1)*parameters->target_number_of_descriptors));
    }

    //ds obtain mean of relative number of matches
    const double mean_completeness_balanced    = getMeanRelativeNumberOfMatches(hbst_balanced,
                                                                                query_descriptors_total,
                                                                                feasible_number_of_matches_per_query,
                                                                                parameters->maximum_descriptor_distance);
    const double mean_completeness_incremental = getMeanRelativeNumberOfMatches(hbst_incremental,
                                                                                query_descriptors_total,
                                                                                feasible_number_of_matches_per_query,
                                                                                parameters->maximum_descriptor_distance);

    //ds initially we need to compute the resulting mean completeness for our estimate
    if (depth == 1) {
      completeness_first_split = getMeanRelativeNumberOfMatches(query_descriptors_total,
                                                                input_descriptors_total,
                                                                feasible_number_of_matches_per_query,
                                                                parameters->maximum_descriptor_distance);
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
    cv::Mat image = cv::imread(image_with_pose->file_path, CV_LOAD_IMAGE_GRAYSCALE);

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

    //ds cull descriptors
    descriptors = descriptors(cv::Rect(0, 0, descriptors.cols, target_number_of_descriptors_per_image_));

    //ds obtain matchables for each descriptor with continuous indexing
    std::vector<uint64_t> indices(descriptors.rows, 0);
    std::for_each(indices.begin(), indices.end(), [](uint64_t &index){++index;});
    MatchableVector matchables_current(Tree::getMatchables(descriptors, indices, image_with_pose->image_number));
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

const double getMeanRelativeNumberOfMatches(const MatchableVector& query_descriptors_,
                                            const MatchableVector& reference_descriptors_,
                                            const std::map<const Matchable*, uint64_t>& feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_hamming_) {

  //ds accumulated mean completeness for all bits
  double accumulated_mean_completeness = 0;

  //ds for each bit index
  for (uint32_t k = 0; k < number_of_checked_bits_for_mean_guess; ++k) {

    //ds partition the input set according to the checked bit
    MatchableVector reference_descriptors_left;
    MatchableVector reference_descriptors_right;
    for (Matchable* input_descriptor: reference_descriptors_) {
      if (input_descriptor->descriptor[k]) {
        reference_descriptors_right.push_back(input_descriptor);
      } else {
        reference_descriptors_left.push_back(input_descriptor);
      }
    }

    //ds counting
    double relative_number_of_matches_accumulated = 0;

    //ds match all queries in the corresponding leafs to compute the completeness
    for (const Matchable* query_descriptor: query_descriptors_) {
      uint64_t number_of_matches = 0;
      if (query_descriptor->descriptor[k]) {

        //ds match against all descriptors in right leaf
        number_of_matches = getNumberOfMatches(query_descriptor, reference_descriptors_right, maximum_distance_hamming_);
      } else {

        //ds match against all descriptors in left leaf
        number_of_matches = getNumberOfMatches(query_descriptor, reference_descriptors_left, maximum_distance_hamming_);
      }

      //ds update completeness for this descriptor and bit
      const uint64_t feasible_number_of_matches = feasible_number_of_matches_per_query_.at(query_descriptor);
      if (feasible_number_of_matches > 0) {
        relative_number_of_matches_accumulated += static_cast<double>(number_of_matches)/feasible_number_of_matches;
      } else {
        relative_number_of_matches_accumulated += 1;
      }
    }
    accumulated_mean_completeness += relative_number_of_matches_accumulated/query_descriptors_.size();
    std::cerr << "computed mean completeness for bit index: " << k << " with value: " << relative_number_of_matches_accumulated/query_descriptors_.size() << std::endl;
  }

  //ds compute mean completeness over all queries for the current bit index
  return accumulated_mean_completeness/number_of_checked_bits_for_mean_guess;
}
