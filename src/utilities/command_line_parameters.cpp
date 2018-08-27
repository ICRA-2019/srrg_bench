#include "command_line_parameters.h"

#include "thirdparty/ldahash/ldahash.h"
#include <bitset>

namespace srrg_bench {

CommandLineParameters::~CommandLineParameters() {

  //ds clear augmentation mappings
  for (const std::pair<std::string, BinaryStringGrid*>& mapping: mappings_image_coordinates_to_augmentation) {
    delete mapping.second;
  }
}

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
    } else if (!std::strcmp(argv_[c], "-table-number") || !std::strcmp(argv_[c], "-tn")) {
      c++; if (c == argc_) {break;}
      table_number = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-depth")) {
      c++; if (c == argc_) {break;}
      maximum_depth = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-display-scale") || !std::strcmp(argv_[c], "-ds")) {
      c++; if (c == argc_) {break;}
      display_scale = std::stod(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-timestamps")) {
      c++; if (c == argc_) {break;}
      file_name_image_timestamps = argv_[c];
    } else if (!std::strcmp(argv_[c], "-images-query")) {
      c++; if (c == argc_) {break;}
      folder_images = argv_[c];
    } else if (!std::strcmp(argv_[c], "-images-reference")) {
      c++; if (c == argc_) {break;}
      folder_images_cross = argv_[c];
    } else if (!std::strcmp(argv_[c], "-position-augmentation")) {
      c++; if (c == argc_) {break;}
      number_of_augmentation_bins_horizontal = std::stoi(argv_[c]);
      c++; if (c == argc_) {break;}
      number_of_augmentation_bins_vertical = std::stoi(argv_[c]);
      c++; if (c == argc_) {break;}
      augmentation_weight = std::stoi(argv_[c]);
    } else if (!std::strcmp(argv_[c], "-semantic-augmentation")) {
      semantic_augmentation = true;
      c++; if (c == argc_) {break;}
      augmentation_weight = std::stoi(argv_[c]);
    }
    c++;
  }
}

void CommandLineParameters::validate(std::ostream& stream_) {
  if (folder_images.empty()) {
    stream_ << "ERROR: no images specified (use -images <folder_images>)" << std::endl;
    throw std::runtime_error("");
  }
  if (file_name_poses_ground_truth.empty() &&
      parsing_mode != "zubud"              &&
      parsing_mode != "oxford"             &&
      parsing_mode != "paris"              &&
      parsing_mode != "holidays"           ) {
    stream_ << "ERROR: no poses specified (use -poses <poses_gt>)" << std::endl;
    throw std::runtime_error("");
  }
  if (file_name_closures_ground_truth == "" && (parsing_mode != "oxford" || parsing_mode != "paris")) {
    stream_ << "WARNING: no closures ground truth specified (use -closures <closures_gt>) - computing full feasibility map" << std::endl;
  }
  if (file_name_closures_ground_truth != ""                                      &&
      file_name_closures_ground_truth.find(descriptor_type) == std::string::npos &&
      file_name_closures_ground_truth.find("SIFT") == std::string::npos          &&
      parsing_mode != "zubud"                                                    &&
      parsing_mode != "oxford"                                                   &&
      parsing_mode != "paris"                                                    &&
      parsing_mode != "holidays"                                                 ) {
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
  if (parsing_mode == "zubud") {

    //ds check if files are missing
    if (folder_images.empty() || folder_images_cross.empty() || file_name_closures_ground_truth.empty()) {
      stream_ << "ERROR: insufficient data provided" << std::endl;
      throw std::runtime_error("");
    }
  }
  if (parsing_mode == "paris") {

    //ds check if files are missing
    if (folder_images.empty() || folder_images_cross.empty()) {
      stream_ << "ERROR: insufficient data provided" << std::endl;
      throw std::runtime_error("");
    }
  }

  //ds if position augmentation is desired
  if (number_of_augmentation_bins_horizontal > 0 && number_of_augmentation_bins_vertical > 0) {
    number_of_augmented_bits = number_of_augmentation_bins_horizontal+number_of_augmentation_bins_vertical-2;

    //ds check augmentation bits
    if (number_of_augmented_bits%8 != 0) {
      stream_ << "ERROR: choose bytewise augmentation (multiples of 8)" << std::endl;
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
  if (number_of_augmentation_bins_horizontal > 0 && number_of_augmentation_bins_vertical > 0) {
    WRITE_VARIABLE(stream_, number_of_augmentation_bins_horizontal);
    WRITE_VARIABLE(stream_, number_of_augmentation_bins_vertical);
    WRITE_VARIABLE(stream_, number_of_augmented_bits);
    WRITE_VARIABLE(stream_, augmentation_weight);
  }
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
  } else if (parsing_mode == "zubud" || parsing_mode == "paris") {
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
    } else if (!file_name_poses_ground_truth.empty()) {
      evaluator->loadImagesWithPosesFromFileOxford(file_name_poses_ground_truth, folder_images);
    } else {
      evaluator->loadImagesFromDirectoryOxford(folder_images, folder_images_cross, "oxford");
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
  } else if (parsing_mode == "zubud") {
    evaluator->loadImagesFromDirectoryZubud(folder_images, folder_images_cross);
  } else if (parsing_mode == "paris") {
    evaluator->loadImagesFromDirectoryOxford(folder_images, folder_images_cross, "paris");
  } else if (parsing_mode == "holidays") {
    evaluator->loadImagesFromDirectoryHolidays(folder_images, file_name_closures_ground_truth);
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
  if (file_name_closures_ground_truth.empty() || parsing_mode == "holidays") {

    //ds nothing to do for oxford and paris
    if (parsing_mode == "oxford" || parsing_mode == "paris" || parsing_mode == "holidays") {

      //ds do nothing

    } else {

      //ds compute feasibility
      evaluator->computeLoopClosureFeasibilityMap(image_number_start,
                                                  image_number_stop,
                                                  query_interspace,
                                                  maximum_difference_position_meters,
                                                  maximum_difference_angle_radians,
                                                  minimum_distance_between_closure_images);
    }
  } else if (parsing_mode == "zubud") {

    //ds compute closures from mapping
    evaluator->computeLoopClosureFeasibilityMap(file_name_closures_ground_truth, '\t');

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
    descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTOR_SIZE_BYTES); //128, 256, 512 bits
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "orb") {
    feature_detector     = cv::ORB::create(1.25*target_number_of_descriptors);
    descriptor_extractor = cv::ORB::create(); //256 bits
    distance_norm        = cv::NORM_HAMMING;
  } else if (descriptor_type == "brisk") {
    feature_detector     = cv::BRISK::create(2*fast_detector_threshold);
    descriptor_extractor = cv::BRISK::create(); //512 bits
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
    descriptor_extractor = cv::xfeatures2d::SIFT::create(target_number_of_descriptors); //ds 128 floats (4096 bits)
    distance_norm        = cv::NORM_L2;
  } else if (descriptor_type == "bold") {
    feature_detector        = cv::xfeatures2d::HarrisLaplaceFeatureDetector::create();
    bold_descriptor_handler = std::make_shared<BOLD>(); //ds 512 bits
  } else if (descriptor_type == "ldahash") {
    //ds nothing needed //ds 128 bits
  }
    else {
    stream_ << "ERROR: unknown descriptor type: " << descriptor_type << std::endl;
    throw std::runtime_error("");
  }
#endif

  //ds for nordland we use the GFTT for an even cross-season feature distribution
  if (parsing_mode == "nordland") {
    feature_detector = cv::GFTTDetector::create(1.5*target_number_of_descriptors, 1e-3, 10);
  }
}

void CommandLineParameters::computeDescriptors(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& descriptors_, const bool sort_keypoints_by_response_) {

  //ds check for custom descriptors TODO sorting
  if (descriptor_type == "bold") {

    //ds detect keypoints
    feature_detector->detect(image_, keypoints_);

    //ds kept keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints_kept(0);
    std::vector<cv::Mat> descriptors(0);

    //ds for each keypoint patch
    for (cv::KeyPoint& keypoint: keypoints_) {
      const int32_t patch_size      = 32;
      const int32_t patch_size_half = patch_size/2;

      //ds if computation is possible
      if (keypoint.pt.x-patch_size_half > 0           &&
          keypoint.pt.x+patch_size_half < image_.cols &&
          keypoint.pt.y-patch_size_half > 0           &&
          keypoint.pt.y+patch_size_half < image_.rows ) {
        keypoint.size = patch_size;

        //ds compute descriptor and mask
        cv::Mat descriptor, mask;
        cv::Mat patch(image_(cv::Rect2i(keypoint.pt.x-patch_size_half,
                                        keypoint.pt.y-patch_size_half,
                                        patch_size,
                                        patch_size)));
        bold_descriptor_handler->compute_patch(patch, descriptor, mask);

        //ds keep keypoint and descriptor
        keypoints_kept.push_back(keypoint);
        descriptors.push_back(descriptor);
      }
    }

    //ds set output
    keypoints_   = keypoints_kept;
    descriptors_ = cv::Mat(keypoints_.size(), descriptors.front().cols, descriptors.front().type());
    for (uint32_t u = 0; u < descriptors.size(); ++u) {
      descriptors_.row(u) = descriptors[u];
    }
  } else if (descriptor_type == "ldahash") {

    //ds call modified ldahash detection and computation method TODO sorting
    run_sifthash(image_, DIF128, keypoints_, descriptors_);
  } else {

    //ds detect keypoints
    feature_detector->detect(image_, keypoints_);

    //ds sort keypoints descendingly by response value (for culling after descriptor computation)
    if (sort_keypoints_by_response_) {
      std::sort(keypoints_.begin(), keypoints_.end(), [](const cv::KeyPoint& a_, const cv::KeyPoint& b_){return a_.response > b_.response;});
    }

    //ds compute descriptors
    descriptor_extractor->compute(image_, keypoints_, descriptors_);
  }

  //ds check if augmentation is desired
  if (number_of_augmented_bits > 0 && augmentation_weight > 0) {

    //ds image resolution key (to obtain fixed mapping)
    const std::string key = std::to_string(image_.rows)+"x"+std::to_string(image_.cols);

    //ds check if augmentations are not initialized yet for this image resolution
    if (number_of_image_rows != static_cast<uint32_t>(image_.rows) ||
        number_of_image_cols != static_cast<uint32_t>(image_.cols) ) {
      number_of_image_rows = image_.rows;
      number_of_image_cols = image_.cols;

      //ds if mapping is not existing yet - configure a new mapping for this key
      if (mappings_image_coordinates_to_augmentation.count(key) == 0) {
        configurePositionAugmentation(key);
      }
    }

    //ds check augmentation bits
    if (number_of_augmented_bits%8 != 0) {
      throw std::runtime_error("choose bytewise augmentation (multiples of 8)");
    }
    const uint32_t number_of_augmented_bytes = augmentation_weight*number_of_augmented_bits/8;

    //ds reserve augmentented descriptor matrix
    cv::Mat descriptors_augmented = cv::Mat(descriptors_.rows, descriptors_.cols+number_of_augmented_bytes, descriptors_.type());

    //ds obtain mapping (must work at this point)
    BinaryStringGrid* mapping = mappings_image_coordinates_to_augmentation.at(key);

    //ds augment each descriptor
    for (int32_t i = 0; i < descriptors_augmented.rows; ++i) {

      //ds for all bytes
      uint32_t augmentation_bit_index = 0;
      for (int32_t j = 0; j < descriptors_augmented.cols; ++j) {
        if (j < descriptors_.cols) {
          descriptors_augmented.row(i).at<uchar>(j) = descriptors_.row(i).at<uchar>(j);
        } else {

          //ds repeat augmentations until completed (index j keeps moving)
          if (augmentation_bit_index == number_of_augmented_bits) {
            augmentation_bit_index = 0;
          }

          //ds obtain the mapping for the descriptor
          const uint32_t& row = keypoints_[i].pt.y;
          const uint32_t& col = keypoints_[i].pt.x;
          const std::string& augmentation = mapping->at(row,col);

          //ds build uchar bitset and set augmentation to descriptor
          const std::bitset<8> data(augmentation.substr(augmentation_bit_index, 8));
          descriptors_augmented.row(i).at<uchar>(j) = static_cast<uchar>(data.to_ulong());
          augmentation_bit_index += 8;
        }
      }
    }

    //ds overwrite output descriptors
    descriptors_ = descriptors_augmented;
  }
}

void CommandLineParameters::computeDescriptors(const cv::Mat& image_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& descriptors_, const uint32_t& target_number_of_descriptors_) {

  //ds compute descriptors after sorting detected keypoints
  computeDescriptors(image_, keypoints_, descriptors_, true);

  //ds rebuild descriptor matrix and keypoints vector unless for specific descriptors
  if (descriptor_type != "ldahash"  &&
      descriptor_type != "binboost" &&
      descriptor_type != "bold"     ) {

    //ds check insufficient descriptor number
    if (keypoints_.size() < target_number_of_descriptors_) {
      std::cerr << "\nWARNING: insufficient number of descriptors computed: " << keypoints_.size()
                << " < " << target_number_of_descriptors_ << ", adjust keypoint detector threshold" << std::endl;
      return;
    }

    keypoints_.resize(target_number_of_descriptors_);
    descriptors_ = descriptors_(cv::Rect(0, 0, descriptors_.cols, target_number_of_descriptors_));
  }
}

void CommandLineParameters::configurePositionAugmentation(const std::string& image_resolution_key_) {
  if (number_of_image_rows == 0                   ||
      number_of_image_cols == 0                   ||
      number_of_augmentation_bins_horizontal == 0 ||
      number_of_augmentation_bins_vertical == 0   ||
      number_of_augmented_bits == 0               ||
      augmentation_weight == 0                    ||
      semantic_augmentation                       ) {
    return;
  }

  //ds we compute a binary string mapping for each pixel [r,c] of the image for fast access
  BinaryStringGrid* mapping = new BinaryStringGrid(number_of_image_rows, number_of_image_cols, number_of_augmented_bits);

  //ds compute average bin widths in pixels
  const uint32_t bin_width_row_pixels = std::ceil(static_cast<double>(number_of_image_rows)/number_of_augmentation_bins_vertical);
  const uint32_t bin_width_col_pixels = std::ceil(static_cast<double>(number_of_image_cols)/number_of_augmentation_bins_horizontal);
  const uint32_t augmented_bits_in_cols = number_of_augmentation_bins_horizontal-1;

  //ds build augmentation map
  uint32_t bin_index_row = 0;
  for (uint32_t row = 0; row < number_of_image_rows; ++row) {

    //ds check if we have to move to the next row bin
    if (row > (bin_index_row+1)*bin_width_row_pixels) {
      ++bin_index_row;
    }

    //ds check complete row
    uint32_t bin_index_col = 0;
    for (uint32_t col = 0; col < number_of_image_cols; ++col) {

      //ds check if we have to move to the next col bin
      if (col > (bin_index_col+1)*bin_width_col_pixels) {
        ++bin_index_col;
      }

      //ds compute binary string for rows (we prefix them to the cols)
      for (uint32_t bit_index_to_set = 0; bit_index_to_set < bin_index_row; ++bit_index_to_set) {
        mapping->at(row,col)[bit_index_to_set] = '1';
      }

      //ds compute binary string for cols
      for (uint32_t bit_index_to_set = 0; bit_index_to_set < bin_index_col; ++bit_index_to_set) {
        mapping->at(row,col)[bit_index_to_set+augmented_bits_in_cols] = '1';
      }
    }
  }

  std::cerr << "configurePositionAugmentation|created mapping: " << number_of_augmentation_bins_horizontal << "x" << number_of_augmentation_bins_vertical
            << " with key: " << image_resolution_key_ << std::endl;
  for (uint32_t row = 0; row < number_of_augmentation_bins_vertical; ++row) {
    for (uint32_t col = 0; col < number_of_augmentation_bins_horizontal; ++col) {
      std::cerr << mapping->at(row*bin_width_row_pixels+1,col*bin_width_col_pixels+1) << " ";
    }
    std::cerr << std::endl;
  }

  //ds set mapping to key
  mappings_image_coordinates_to_augmentation.insert(std::make_pair(image_resolution_key_, mapping));
}

cv::Mat CommandLineParameters::readImage(const std::string& image_file_path_) const {
  cv::Mat image = cv::imread(image_file_path_, CV_LOAD_IMAGE_GRAYSCALE);

  //ds floor images dimensions by 10% to have not infinite different dimensions
  if (parsing_mode == "oxford" || parsing_mode == "paris") {
    const int32_t rows_cropped = (image.rows/10)*10;
    const int32_t cols_cropped = (image.cols/10)*10;
    image = image(cv::Rect((image.cols-cols_cropped)/2.0, (image.rows-rows_cropped)/2.0, cols_cropped, rows_cropped));
  }

  return image;
}

void CommandLineParameters::displayKeypoints(const cv::Mat& image_,
                                             const std::vector<cv::KeyPoint>& keypoints_) const {
  if (use_gui) {
    cv::Mat image_display = image_;
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
    for (const cv::KeyPoint& keypoint: keypoints_) {
      cv::circle(image_display, keypoint.pt, 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(image_display, keypoint.pt, keypoint.size, cv::Scalar(0, 0, 255), 1);
    }

    //ds if position augmentation is set - display grid
    if (number_of_augmentation_bins_horizontal > 0 && number_of_augmentation_bins_vertical > 0) {
      const double pixels_per_horizontal_bin = static_cast<double>(image_display.cols)/number_of_augmentation_bins_horizontal;
      const double pixels_per_vertical_bin   = static_cast<double>(image_display.rows)/number_of_augmentation_bins_vertical;
      for (uint32_t u = 0; u < number_of_augmentation_bins_horizontal; ++u) {
        for (uint32_t v = 0; v < number_of_augmentation_bins_vertical; ++v) {
          cv::line(image_display,
                   cv::Point2i(u*pixels_per_horizontal_bin, v*pixels_per_vertical_bin),
                   cv::Point2i(u*pixels_per_horizontal_bin, (v+1)*pixels_per_vertical_bin),
                   cv::Scalar(0, 255, 0),
                   1);
          cv::line(image_display,
                   cv::Point2i(u*pixels_per_horizontal_bin, v*pixels_per_vertical_bin),
                   cv::Point2i((u+1)*pixels_per_horizontal_bin, v*pixels_per_vertical_bin),
                   cv::Scalar(0, 255, 0),
                   1);
        }
      }
    }
    cv::imshow("benchmark | "+parsing_mode+" | "+descriptor_type+"-"+std::to_string(DESCRIPTOR_SIZE_BITS+augmentation_weight*number_of_augmented_bits), image_display);
    cv::waitKey(1);
  }
}
}
