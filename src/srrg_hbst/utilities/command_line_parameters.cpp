#include "command_line_parameters.h"

//ds easy logging macro - living from your expressiveness
#define WRITE_VARIABLE(STREAM_, VARIABLE_) \
  STREAM_ << #VARIABLE_ << ": " << VARIABLE_ << std::endl
#define BAR "---------------------------------------------------------------------------------"

namespace srrg_bench {

void CommandLineParameters::parse(const int32_t& argc_, char** argv_) {
  int32_t argc_parsed = 1;
  while(argc_parsed < argc_){
    if (!std::strcmp(argv_[argc_parsed], "-cross")) {
      load_cross_datasets = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-method")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      method_name = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-images")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      folder_images = argv_[argc_parsed];
      if (load_cross_datasets) {
        argc_parsed++; if (argc_parsed == argc_) {break;}
        folder_images_cross = argv_[argc_parsed];
      }
    } else if(!std::strcmp(argv_[argc_parsed], "-poses")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      file_name_poses_ground_truth = argv_[argc_parsed];
      if (load_cross_datasets) {
        argc_parsed++; if (argc_parsed == argc_) {break;}
        file_name_poses_ground_truth_cross = argv_[argc_parsed];
      }
    } else if(!std::strcmp(argv_[argc_parsed], "-closures")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      file_name_closures_ground_truth = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-mode")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      parsing_mode = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-t")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      fast_detector_threshold = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-m")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      maximum_distance_hamming = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-r")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      target_recall = std::stod(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-space")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      query_interspace = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-distance")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      minimum_distance_between_closure_images = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-descriptor")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      if (method_name.compare("bow") == 0) {std::cerr << "WARNING: -descriptor is parameter ignored (bow)" << std::endl;}
      descriptor_type = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-voc")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      file_path_vocabulary = argv_[argc_parsed];
    } else if (!std::strcmp(argv_[argc_parsed], "-use-gui") || !std::strcmp(argv_[argc_parsed], "-ug")) {
      use_gui = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-score-only") || !std::strcmp(argv_[argc_parsed], "-so")) {
      compute_score_only = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-n")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      target_number_of_descriptors = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-random")) {
      use_random_splitting = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-samples")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      number_of_samples = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-uneven")) {
      use_uneven_splitting = true;
    } else if (!std::strcmp(argv_[argc_parsed], "-delay")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      training_delay_in_frames = std::stoi(argv_[argc_parsed]);
    } else if (!std::strcmp(argv_[argc_parsed], "-leaf-size") || !std::strcmp(argv_[argc_parsed], "-ls")) {
      argc_parsed++; if (argc_parsed == argc_) {break;}
      maximum_leaf_size = std::stoi(argv_[argc_parsed]);
    }
    argc_parsed++;
  }
}

void CommandLineParameters::validate(std::ostream& stream_) {
  if (folder_images.empty()) {
    stream_ << "ERROR: no images specified (use -images <folder_images>)" << std::endl;
    throw std::runtime_error("");
  }
  if (file_name_poses_ground_truth.empty()) {
    stream_ << "ERROR: no poses specified (use -poses <poses_gt>)" << std::endl;
    throw std::runtime_error("");
  }
  if (file_name_closures_ground_truth == "") {
    stream_ << "WARNING: no closures ground truth specified (use -closures <closures_gt>) - computing full feasibility map" << std::endl;
  }
  if (file_name_closures_ground_truth != "" && file_name_closures_ground_truth.find(descriptor_type) == std::string::npos) {
    stream_ << "ERROR: invalid descriptor type in closures ground truth: " << file_name_closures_ground_truth << std::endl;
    throw std::runtime_error("");
  }
  if (parsing_mode == "lucia" && file_name_image_timestamps.length() == 0) {
    stream_ << "ERROR: no image UQ St Lucia timestamps file specified (use -timestamps <image_timestamps>)" << std::endl;
    throw std::runtime_error("");
  }
  if (method_name == "bow" && file_path_vocabulary.empty()) {
    stream_ << "ERROR: no vocabulary provided (use -voc <file_descriptor_vocabulary>)" << std::endl;
    throw std::runtime_error("");
  }

  //ds for dbow2 the descriptor type must be set in a hardcoded manner - hence we have to overwrite it here
  if (method_name == "bow") {
#if DBOW2_DESCRIPTOR_TYPE == 0
    descriptor_type = "brief";
#elif DBOW2_DESCRIPTOR_TYPE == 1
    descriptor_type = "orb";
#else
    stream_ << "ERROR: bow descriptor type not set" << std::endl;
    throw std::runtime_error("");
#endif
  }
}

void CommandLineParameters::write(std::ostream& stream_) {
  stream_ << BAR << std::endl;
  WRITE_VARIABLE(stream_, method_name);
  WRITE_VARIABLE(stream_, folder_images);
  WRITE_VARIABLE(stream_, file_name_poses_ground_truth);
  WRITE_VARIABLE(stream_, file_name_closures_ground_truth);
  WRITE_VARIABLE(stream_, parsing_mode);
  stream_ << BAR << std::endl;
  WRITE_VARIABLE(stream_, query_interspace);
  WRITE_VARIABLE(stream_, minimum_distance_between_closure_images);
  WRITE_VARIABLE(stream_, maximum_difference_position_meters);
  WRITE_VARIABLE(stream_, maximum_difference_angle_radians);
  WRITE_VARIABLE(stream_, load_cross_datasets);
  stream_ << BAR << std::endl;
  WRITE_VARIABLE(stream_, descriptor_type);
  WRITE_VARIABLE(stream_, DESCRIPTOR_SIZE_BYTES);
  WRITE_VARIABLE(stream_, DESCRIPTOR_SIZE_BITS);
  WRITE_VARIABLE(stream_, maximum_distance_hamming);
  WRITE_VARIABLE(stream_, fast_detector_threshold);
  WRITE_VARIABLE(stream_, use_gui);
  WRITE_VARIABLE(stream_, target_number_of_descriptors);
  WRITE_VARIABLE(stream_, target_recall);
  WRITE_VARIABLE(stream_, number_of_samples);
  stream_ << BAR << std::endl;
  if (method_name == "hbst") {
    WRITE_VARIABLE(stream_, maximum_leaf_size);
    WRITE_VARIABLE(stream_, maximum_partitioning);
    WRITE_VARIABLE(stream_, use_random_splitting);
    WRITE_VARIABLE(stream_, use_uneven_splitting);
  } else if (method_name == "bow") {
    WRITE_VARIABLE(stream_, file_path_vocabulary);
    WRITE_VARIABLE(stream_, use_direct_index);
    WRITE_VARIABLE(stream_, direct_index_levels);
    WRITE_VARIABLE(stream_, compute_score_only);
  } else if (method_name == "flannlsh") {
    WRITE_VARIABLE(stream_, table_number);
    WRITE_VARIABLE(stream_, key_size);
    WRITE_VARIABLE(stream_, multi_probe_level);
  }
  if (parsing_mode == "lucia") {
    stream_ << BAR << std::endl;
    WRITE_VARIABLE(stream_, file_name_image_timestamps);
  } else if (parsing_mode == "oxford") {
    stream_ << BAR << std::endl;
    WRITE_VARIABLE(stream_, file_name_poses_ground_truth_cross);
    WRITE_VARIABLE(stream_, folder_images_cross);
  }
  stream_ << BAR << std::endl;
}

void CommandLineParameters::configure(std::ostream& stream_) {
  evaluator = std::make_shared<LoopClosureEvaluator>();

  //ds load ground truth poses - depending on chosen parsing mode
  if (parsing_mode == "kitti") {
    evaluator->loadImagesWithPosesFromFileKITTI(file_name_poses_ground_truth, folder_images);
  } else if (parsing_mode == "malaga") {
    evaluator->loadImagesWithPosesFromFileMalaga(file_name_poses_ground_truth, folder_images);

    //ds adjust thresholds for malaga precision (GPS only)
    maximum_difference_position_meters *= 2;
    maximum_difference_angle_radians   *= 2;
  } else if (parsing_mode == "lucia") {
    evaluator->loadImagesWithPosesFromFileLucia(file_name_poses_ground_truth, file_name_image_timestamps);

    //ds adjust detector threshold to high image resolution
    fast_detector_threshold *= 3;
  } else if (parsing_mode == "oxford") {
    if (load_cross_datasets) {
      evaluator->loadImagesWithPosesFromFileOxford(file_name_poses_ground_truth,
                                                   folder_images,
                                                   file_name_poses_ground_truth_cross,
                                                   folder_images_cross);
    } else {
      evaluator->loadImagesWithPosesFromFileOxford(file_name_poses_ground_truth, folder_images);
    }
  } else {
    stream_ << "ERROR: unknown selected parsing mode: '" << parsing_mode << "'" << std::endl;
    throw std::runtime_error("");
  }

  //ds compute all feasible closures for the given interspace - check if no closure ground truth file is provided
  if (file_name_closures_ground_truth == "") {

    //ds compute feasibility
    evaluator->computeLoopClosureFeasibilityMap(query_interspace,
                                                maximum_difference_position_meters,
                                                maximum_difference_angle_radians,
                                                minimum_distance_between_closure_images);
  } else {

    //ds load feasible closures (BF filtered) from a file
    evaluator->loadClosures(file_name_closures_ground_truth,
                            query_interspace,
                            maximum_difference_position_meters,
                            maximum_difference_angle_radians,
                            minimum_distance_between_closure_images);
  }

  //ds load default keypoint detector
#if CV_MAJOR_VERSION == 2
  feature_detector     = new cv::FastFeatureDetector(fast_detector_threshold);
#elif CV_MAJOR_VERSION == 3
  feature_detector = cv::FastFeatureDetector::create(fast_detector_threshold);
#endif

//ds chose descriptor extractor and matching keypoint detector if available TODO validate opencv2 compatibility
#if CV_MAJOR_VERSION == 2
  if (descriptor_type == "brief") {
    descriptor_extractor = new cv::BriefDescriptorExtractor(DESCRIPTOR_SIZE_BYTES);
  } else if (descriptor_type == "orb") {
    descriptor_extractor = new cv::ORB();
  } else if (descriptor_type == "brisk") {
    descriptor_extractor = new cv::BRISK();
  } else if (descriptor_type == "freak") {
    descriptor_extractor = new cv::FREAK(); //512 bits
  } else {
    stream_ << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    throw std::runtime_error("");
  }
#elif CV_MAJOR_VERSION == 3
  if (descriptor_type == "brief") {
    descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTOR_SIZE_BYTES);
  } else if (descriptor_type == "orb") {
    feature_detector     = cv::ORB::create(2*target_number_of_descriptors);
    descriptor_extractor = cv::ORB::create();
  } else if (descriptor_type == "brisk") {
    feature_detector     = cv::BRISK::create(fast_detector_threshold);
    descriptor_extractor = cv::BRISK::create();
  } else if (descriptor_type == "freak") {
    descriptor_extractor = cv::xfeatures2d::FREAK::create(); //512 bits
  } else if (descriptor_type == "akaze") {
    feature_detector     = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, fast_detector_threshold/1e4); //486 bits
    descriptor_extractor = cv::AKAZE::create(); //486 bits
  } else {
    stream_ << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    throw std::runtime_error("");
  }
#endif
}
}
