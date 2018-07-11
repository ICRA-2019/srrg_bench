#include "command_line_parameters.h"

namespace srrg_bench {

void CommandLineParameters::parse(const int32_t& argc_, char** argv_) {
  int32_t c = 1;
  while(c < argc_){
    if (!std::strcmp(argv_[c], "-cross")) {
      load_cross_datasets = true;
    } else if (!std::strcmp(argv_[c], "-method")) {
      c++; if (c == argc_) {break;}
      method_name = argv_[c];
    } else if (!std::strcmp(argv_[c], "-images")) {
      c++; if (c == argc_) {break;}
      folder_images = argv_[c];
      if (load_cross_datasets) {
        c++; if (c == argc_) {break;}
        folder_images_cross = argv_[c];
      }
    } else if(!std::strcmp(argv_[c], "-start")) {
      c++; if (c == argc_) {break;}
      image_number_start = std::stoi(argv_[c]);
    } else if(!std::strcmp(argv_[c], "-stop")) {
      c++; if (c == argc_) {break;}
      image_number_stop = std::stoi(argv_[c]);
    } else if(!std::strcmp(argv_[c], "-poses")) {
      c++; if (c == argc_) {break;}
      file_name_poses_ground_truth = argv_[c];
      if (load_cross_datasets) {
        c++; if (c == argc_) {break;}
        file_name_poses_ground_truth_cross = argv_[c];
      }
    } else if(!std::strcmp(argv_[c], "-closures")) {
      c++; if (c == argc_) {break;}
      file_name_closures_ground_truth = argv_[c];
    } else if (!std::strcmp(argv_[c], "-mode")) {
      c++; if (c == argc_) {break;}
      parsing_mode = argv_[c];
    } else if (!std::strcmp(argv_[c], "-threads")) {
      c++; if (c == argc_) {break;}
      number_of_openmp_threads = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-t")) {
      c++; if (c == argc_) {break;}
      fast_detector_threshold = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-m")) {
      c++; if (c == argc_) {break;}
      maximum_descriptor_distance = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-r")) {
      c++; if (c == argc_) {break;}
      target_recall = std::stod(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-space")) {
      c++; if (c == argc_) {break;}
      query_interspace = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-distance")) {
      c++; if (c == argc_) {break;}
      minimum_distance_between_closure_images = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-descriptor")) {
      c++; if (c == argc_) {break;}
      if (method_name.compare("bow") == 0) {std::cerr << "WARNING: -descriptor is parameter ignored (bow)" << std::endl;}
      descriptor_type = argv_[c];
    } else if (!std::strcmp(argv_[c], "-voc")) {
      c++; if (c == argc_) {break;}
      file_path_vocabulary = argv_[c];
    } else if (!std::strcmp(argv_[c], "-use-gui") || !std::strcmp(argv_[c], "-ug")) {
      use_gui = true;
    } else if (!std::strcmp(argv_[c], "-score-only") || !std::strcmp(argv_[c], "-so")) {
      compute_score_only = true;
    } else if (!std::strcmp(argv_[c], "-n")) {
      c++; if (c == argc_) {break;}
      target_number_of_descriptors = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-random")) {
      use_random_splitting = true;
    } else if (!std::strcmp(argv_[c], "-samples")) {
      c++; if (c == argc_) {break;}
      number_of_samples = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-uneven")) {
      use_uneven_splitting = true;
    } else if (!std::strcmp(argv_[c], "-delay")) {
      c++; if (c == argc_) {break;}
      training_delay_in_frames = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-leaf-size") || !std::strcmp(argv_[c], "-ls")) {
      c++; if (c == argc_) {break;}
      maximum_leaf_size = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-multi-probe-level") || !std::strcmp(argv_[c], "-mpl")) {
      c++; if (c == argc_) {break;}
      multi_probe_level = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-hash-key-length") || !std::strcmp(argv_[c], "-hkl")) {
      c++; if (c == argc_) {break;}
      hash_key_size = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-depth")) {
      c++; if (c == argc_) {break;}
      maximum_depth = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-display-scale") || !std::strcmp(argv_[c], "-ds")) {
      c++; if (c == argc_) {break;}
      display_scale = std::stod(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-timestamps")) {
      c++; if (c == argc_) {break;}
      file_name_image_timestamps = argv_[c];
    }
    c++;
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
  if (file_name_closures_ground_truth != ""                                      &&
      file_name_closures_ground_truth.find(descriptor_type) == std::string::npos &&
      file_name_closures_ground_truth.find("SIFT") == std::string::npos          ) {
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
  if (parsing_mode == "nordland") {

    //ds check if files are missing
    if (folder_images.empty() || folder_images_cross.empty()) {
      stream_ << "ERROR: no video streams provided (use -cross -images <video_query> <video_reference>)" << std::endl;
      throw std::runtime_error("");
    }
    if (file_name_poses_ground_truth.empty() || file_name_poses_ground_truth_cross.empty()) {
      stream_ << "ERROR: no GPS ground truth provided (use -poses <gps_query.csv> <gps_reference.csv>)" << std::endl;
      throw std::runtime_error("");
    }
  }
}

void CommandLineParameters::write(std::ostream& stream_) {
  stream_ << BAR << std::endl;
  WRITE_VARIABLE(stream_, method_name);
  WRITE_VARIABLE(stream_, folder_images);
  WRITE_VARIABLE(stream_, file_name_poses_ground_truth);
  WRITE_VARIABLE(stream_, file_name_closures_ground_truth);
  WRITE_VARIABLE(stream_, parsing_mode);
  WRITE_VARIABLE(stream_, image_number_start);
  WRITE_VARIABLE(stream_, image_number_stop);
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
  WRITE_VARIABLE(stream_, maximum_descriptor_distance);
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
    WRITE_VARIABLE(stream_, maximum_depth);
    stream_ << BAR << std::endl;
  } else if (method_name == "bow") {
    WRITE_VARIABLE(stream_, file_path_vocabulary);
    WRITE_VARIABLE(stream_, use_direct_index);
    WRITE_VARIABLE(stream_, direct_index_levels);
    WRITE_VARIABLE(stream_, compute_score_only);
    stream_ << BAR << std::endl;
  } else if (method_name == "flannlsh") {
    WRITE_VARIABLE(stream_, table_number);
    WRITE_VARIABLE(stream_, hash_key_size);
    WRITE_VARIABLE(stream_, multi_probe_level);
    stream_ << BAR << std::endl;
  }
  if (parsing_mode == "lucia") {
    WRITE_VARIABLE(stream_, file_name_image_timestamps);
    stream_ << BAR << std::endl;
  } else if (parsing_mode == "oxford") {
    WRITE_VARIABLE(stream_, file_name_poses_ground_truth_cross);
    WRITE_VARIABLE(stream_, folder_images_cross);
    stream_ << BAR << std::endl;
  } else if (parsing_mode == "nordland") {
    WRITE_VARIABLE(stream_, file_name_poses_ground_truth_cross);
    WRITE_VARIABLE(stream_, folder_images_cross);
    stream_ << BAR << std::endl;
  }
}

void CommandLineParameters::configure(std::ostream& stream_) {
  evaluator = std::make_shared<LoopClosureEvaluator>();

  //ds load ground truth poses - depending on chosen parsing mode
  if (parsing_mode == "kitti") {
    evaluator->loadImagesWithPosesFromFileKITTI(file_name_poses_ground_truth, folder_images);
  } else if (parsing_mode == "malaga") {
    evaluator->loadImagesWithPosesFromFileMalaga(file_name_poses_ground_truth, folder_images);

    //ds raise thresholds to coarse malaga precision (GPS only)
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
  } else if (parsing_mode == "nordland") {

    //ds open streams and exit on failure
    if (!video_player_query.open(folder_images, cv::CAP_FFMPEG)) {
      stream_ << "ERROR: unable to open video: " << folder_images << std::endl;
      throw std::runtime_error("");
    }
    if (!video_player_reference.open(folder_images_cross, cv::CAP_FFMPEG)) {
      stream_ << "ERROR: unable to open video: " << folder_images_cross << std::endl;
      throw std::runtime_error("");
    }

    //ds ground truth loading (although we know that the image streams are synchronized by location)
    evaluator->loadImagesWithPosesFromFileNordland(file_name_poses_ground_truth,
                                                   folder_images,
                                                   file_name_poses_ground_truth_cross,
                                                   folder_images_cross);
  } else {
    stream_ << "ERROR: unknown selected parsing mode: '" << parsing_mode << "'" << std::endl;
    throw std::runtime_error("");
  }

  //ds update stop number of images to maximum possible if not set
  if (image_number_stop == 0) {
    image_number_stop = evaluator->numberOfImages()-1;
  }

  //ds compute total number of images to process (considering the interspace)
  number_of_images_to_process = static_cast<double>(image_number_stop-image_number_start)/query_interspace;

  //ds compute all feasible closures for the given interspace - check if no closure ground truth file is provided
  if (file_name_closures_ground_truth == "") {

    //ds compute feasibility
    evaluator->computeLoopClosureFeasibilityMap(image_number_start,
                                                image_number_stop,
                                                query_interspace,
                                                maximum_difference_position_meters,
                                                maximum_difference_angle_radians,
                                                minimum_distance_between_closure_images);
  } else {

    //ds load feasible closures (BF filtered) from a file
    evaluator->loadClosures(file_name_closures_ground_truth,
                            image_number_start,
                            image_number_stop,
                            query_interspace,
                            maximum_difference_position_meters,
                            maximum_difference_angle_radians,
                            minimum_distance_between_closure_images);
  }

//ds chose descriptor extractor and matching keypoint detector if available
#if CV_MAJOR_VERSION == 2
  if (descriptor_type == "brief") {
    feature_detector     = new cv::FastFeatureDetector(fast_detector_threshold);
    descriptor_extractor = new cv::BriefDescriptorExtractor(DESCRIPTOR_SIZE_BYTES);
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "orb") {
    feature_detector     = new cv::FastFeatureDetector(fast_detector_threshold);
    descriptor_extractor = new cv::ORB();
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "brisk") {
    feature_detector     = new cv::FastFeatureDetector(fast_detector_threshold);
    descriptor_extractor = new cv::BRISK();
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "freak") {
    feature_detector     = new cv::FastFeatureDetector(fast_detector_threshold);
    descriptor_extractor = new cv::FREAK(); //512 bits
    distance_norm        = cv::NORM_HAMMING;
  } else {
    stream_ << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    throw std::runtime_error("");
  }
#elif CV_MAJOR_VERSION == 3
  if (descriptor_type == "brief") {
    feature_detector     = cv::FastFeatureDetector::create(fast_detector_threshold);
    descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTOR_SIZE_BYTES);
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "orb") {
    feature_detector     = cv::ORB::create(2*target_number_of_descriptors);
    descriptor_extractor = cv::ORB::create();
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "brisk") {
    feature_detector     = cv::BRISK::create(2*fast_detector_threshold);
    descriptor_extractor = cv::BRISK::create();
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "freak") {
    feature_detector     = cv::FastFeatureDetector::create(fast_detector_threshold);
    descriptor_extractor = cv::xfeatures2d::FREAK::create(); //512 bits
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "akaze") {
    feature_detector     = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, fast_detector_threshold/1e5); //486 bits
    descriptor_extractor = cv::AKAZE::create(); //486 bits
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "sift") {
    feature_detector     = cv::xfeatures2d::SIFT::create(target_number_of_descriptors);
    descriptor_extractor = cv::xfeatures2d::SIFT::create(target_number_of_descriptors);
    distance_norm        = cv::NORM_L2;
  } else {
    stream_ << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    throw std::runtime_error("");
  }
#endif

  //ds for nordland we use the GFTT for an even cross-season feature distribution
  if (parsing_mode == "nordland") {
    feature_detector = cv::GFTTDetector::create(1.5*target_number_of_descriptors, 1e-3, 10);
  }
}
}
