#include <omp.h>
#include "utilities/command_line_parameters.h"
#include "srrg_hbst_types/binary_tree.hpp"



//ds HBST configuration
typedef srrg_hbst::BinaryMatchable<DESCRIPTOR_SIZE_BITS> Matchable;
typedef Matchable::Descriptor Descriptor;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef Node::MatchableVector MatchableVector;
typedef srrg_hbst::BinaryTree<Node> Tree;



void loadMatchables(MatchableVector& matchables_total_,
                    const std::set<const srrg_bench::ImageWithPose*>& images_,
                    const std::shared_ptr<srrg_bench::CommandLineParameters> parameters_);

const uint64_t getNumberOfMatches(const Matchable* query_descriptor_,
                                  const MatchableVector& input_descriptors_,
                                  const uint32_t& maximum_distance_matching_);

const double getMeanRelativeNumberOfMatches(const std::shared_ptr<Tree> tree_,
                                            const MatchableVector& query_descriptors_,
                                            const uint64_t* feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_matching_);

int32_t main(int32_t argc_, char** argv_) {

  //ds validate input
  if (argc_ < 5) {
    std::cerr << "invalid call - please use: ./analyze_completeness_monte_carlo "
                 "-images <images_folder> "
                 "-poses <poses_ground_truth> "
                 "-depth <maximum_depth> "
                 "-samples <number_of_samples> "
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

  //ds configure openmp
  omp_set_dynamic(0);
  omp_set_num_threads(parameters->number_of_openmp_threads);
  #pragma omp parallel
  {
    #pragma omp master
    LOG_VARIABLE(omp_get_num_threads());
  }

  //ds compute all sets of input descriptors
  MatchableVector input_descriptors_total;
  std::cerr << "computing input image descriptors: " << std::endl;
  loadMatchables(input_descriptors_total, parameters->evaluator->validTrainImagesWithPoses(), parameters);
  std::cerr << std::endl;
  LOG_VARIABLE(input_descriptors_total.size());
  const uint64_t number_of_input_images(parameters->evaluator->validTrainImagesWithPoses().size());

  //ds compute all sets of query descriptors
  MatchableVector query_descriptors_total;
  std::cerr << "computing query image descriptors: " << std::endl;
  loadMatchables(query_descriptors_total, parameters->evaluator->validQueryImagesWithPoses(), parameters);
  std::cerr << std::endl;
  LOG_VARIABLE(query_descriptors_total.size());

  //ds for all query descriptors
  std::cerr << "computing total number of feasible matches (C_0=1) N_M in root node for tau: " << parameters->maximum_descriptor_distance
            << " complexity: " << static_cast<double>(query_descriptors_total.size())*input_descriptors_total.size() << std::endl;
  uint64_t* feasible_number_of_matches_per_query = new uint64_t[query_descriptors_total.size()];
  uint64_t total_number_of_matches = 0;
  #pragma omp parallel for reduction(+:total_number_of_matches)
  for (uint64_t index_query = 0; index_query < query_descriptors_total.size(); ++index_query) {
    feasible_number_of_matches_per_query[index_query] = 0;

    //ds current query descriptor
    const Matchable* query_descriptor = query_descriptors_total[index_query];

    //ds match against ALL input descriptors (maximum completeness)
    for (const Matchable* input_descriptor: input_descriptors_total) {

      //ds if the distance is within the threshold
      if ((query_descriptor->descriptor^input_descriptor->descriptor).count() < parameters->maximum_descriptor_distance) {
        ++feasible_number_of_matches_per_query[index_query];
      }
    }
    total_number_of_matches += feasible_number_of_matches_per_query[index_query];
  }
  std::cerr << "average number of matches per query: " << static_cast<double>(total_number_of_matches)/query_descriptors_total.size() << std::endl;

  //ds prepare result file
  const std::string file_name_results = "completeness_monte-carlo_"
                                      + parameters->descriptor_type + "-" + std::to_string(DESCRIPTOR_SIZE_BITS) + "_"
                                      + "tau-" + std::to_string(static_cast<uint32_t>(parameters->maximum_descriptor_distance)) + ".txt";
  std::ofstream result_file(file_name_results, std::ios::trunc);
  result_file << "#SAMPLE_NUMBER #MEAN_COMPLETENESS_DEPTH_0 #MEAN_COMPLETENESS_DEPTH_1 #MEAN_COMPLETENESS_DEPTH_.." << std::endl;
  result_file.close();

  //ds empty tree handle
  std::shared_ptr<Tree> hbst_incremental = std::make_shared<Tree>();

  //ds for each sample
  std::cerr << "starting Monte-Carlo sampling for random split HBST completeness evaluation:" << std::endl;
  for (uint32_t sample_number = 0; sample_number < parameters->number_of_samples; ++sample_number) {
    std::cerr << "sample number: " << sample_number << std::endl;
    result_file.open(file_name_results, std::ios::app);
    result_file << sample_number;
    result_file.close();

    //ds for each feasible maximum depth starting from 0 (verification of trivial result 1)
    for (uint32_t maximum_depth = 0; maximum_depth < parameters->maximum_depth; ++maximum_depth) {

      //ds clear previous structure (without deallocation memory for matchables)
      hbst_incremental->clear(false);

      //ds construct tree with new maximum depth (only constraint)
      Tree::Node::maximum_depth = maximum_depth;

      //ds construct incremental tree - in batches
      for (uint64_t number_of_insertions = 0; number_of_insertions < number_of_input_images; ++number_of_insertions) {
        hbst_incremental->add(MatchableVector(input_descriptors_total.begin()+number_of_insertions*parameters->target_number_of_descriptors,
                                              input_descriptors_total.begin()+(number_of_insertions+1)*parameters->target_number_of_descriptors),
                                              srrg_hbst::SplittingStrategy::SplitRandomUniform);
      }

      //ds compute mean completeness
      const double mean_completeness_incremental = getMeanRelativeNumberOfMatches(hbst_incremental,
                                                                                  query_descriptors_total,
                                                                                  feasible_number_of_matches_per_query,
                                                                                  parameters->maximum_descriptor_distance);

      std::cerr << " - depth: " << maximum_depth
                << " mean completeness: " << mean_completeness_incremental
                << std::endl;

      //ds save result to file (we reopen it in order to not keep a file handle all the time)
      result_file.open(file_name_results, std::ios::app);
      result_file << " " << mean_completeness_incremental ;
      result_file.close();
    }

    //ds complete sample
    result_file.open(file_name_results, std::ios::app);
    result_file << std::endl;
    result_file.close();
  }

  //ds clear tree (without freeing matchables)
  hbst_incremental->clear(false);

  //ds free all matchables
  for (const Matchable* matchable: query_descriptors_total) {
    delete matchable;
  }
  for (const Matchable* matchable: input_descriptors_total) {
    delete matchable;
  }
  delete [] feasible_number_of_matches_per_query;

  //ds done
  return EXIT_SUCCESS;
}

void loadMatchables(MatchableVector& matchables_total_,
                    const std::set<const srrg_bench::ImageWithPose*>& images_,
                    const std::shared_ptr<srrg_bench::CommandLineParameters> parameters_) {
  matchables_total_.clear();
  for (const srrg_bench::ImageWithPose* image_with_pose: images_) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //ds load image
    cv::Mat image = cv::imread(image_with_pose->file_name, CV_LOAD_IMAGE_GRAYSCALE);

    //ds detect keypoints
    parameters_->feature_detector->detect(image, keypoints);

    //ds sort keypoints descendingly by response value (for culling after descriptor computation)
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});

    //ds compute BRIEF descriptors for sorted keypoints
    parameters_->descriptor_extractor->compute(image, keypoints, descriptors);

    //ds check insufficient descriptor number
    if (keypoints.size() < parameters_->target_number_of_descriptors) {
      std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints.size()
                << " < " << parameters_->target_number_of_descriptors << ", adjust detector threshold\n" << std::endl;
      throw std::runtime_error("insufficient number of descriptors computed");
    }

    //ds compute matchables and store them
    MatchableVector matchables_current(Tree::getMatchablesWithIndex(descriptors(cv::Rect(0, 0, descriptors.cols, parameters_->target_number_of_descriptors)),
                                                                    image_with_pose->image_number,
                                                                    matchables_total_.size()));
    matchables_total_.insert(matchables_total_.end(), matchables_current.begin(), matchables_current.end());
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
                                            const uint64_t* feasible_number_of_matches_per_query_,
                                            const uint32_t& maximum_distance_matching_) {

  //ds relative number of matches summed up over all queries
  double relative_number_of_matches_accumulated = 0;

  //ds for each query descriptor
  #pragma omp parallel for reduction(+:relative_number_of_matches_accumulated)
  for (uint64_t index_query = 0; index_query < query_descriptors_.size(); ++index_query) {
    const Matchable* query_descriptor = query_descriptors_[index_query];

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
    const uint64_t feasible_number_of_matches = feasible_number_of_matches_per_query_[query_descriptor->identifier];
    if (feasible_number_of_matches > 0) {
      relative_number_of_matches_accumulated += static_cast<double>(number_of_matches)/feasible_number_of_matches;
    } else {
      relative_number_of_matches_accumulated += 1;
    }
  }
  return relative_number_of_matches_accumulated/query_descriptors_.size();
}
