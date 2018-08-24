#include "loop_closure_evaluator.h"

#include "dirent.h"
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

namespace srrg_bench {

LoopClosureEvaluator::LoopClosureEvaluator() {
  _image_poses_ground_truth.clear();
  _closure_feasability_map.clear();
  _closure_map_bf.clear();
  _valid_query_image_numbers.clear();
  _valid_train_image_numbers.clear();
  _valid_closures.clear();
  _invalid_closures.clear();
}

LoopClosureEvaluator::~LoopClosureEvaluator() {
  for (const ImageWithPose* image_with_pose: _image_poses_ground_truth) {
    delete image_with_pose;
  }
  _image_poses_ground_truth.clear();
  _closure_feasability_map.clear();
  _closure_map_bf.clear();
  _valid_query_image_numbers.clear();
  _valid_train_image_numbers.clear();
  _valid_closures.clear();
  _invalid_closures.clear();
}

void LoopClosureEvaluator::loadImagesWithPosesFromFileKITTI(const std::string& file_name_poses_ground_truth_, const std::string& folder_images_) {
  _trajectory_length_meters = 0;
  _image_poses_ground_truth.clear();

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_poses_ground_truth_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::loadPosesFromFileKITTI|ERROR: unable to open file: " << file_name_poses_ground_truth_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds start parsing all poses to the last line (initialize vector of eigen matrices, avoiding memory alignment issues)
  _image_poses_ground_truth.resize(1e6);
  ImageNumberQuery image_number = 0;
  while (image_number < _image_poses_ground_truth.size()) {

    //ds reading buffers
    std::string line_buffer("");
    std::getline(file_ground_truth, line_buffer);
    std::istringstream stream(line_buffer);
    if (line_buffer.empty()) {
      break;
    }

    //ds parse pose matrix directly from file (KITTI format)
    Eigen::Isometry3d pose(Eigen::Isometry3d::Identity());
    for(uint8_t u = 0; u < 3; ++u) {
      for(uint8_t v = 0; v < 4; ++v) {
        stream >> pose(u,v);
      }
    }

    //ds generate file name
    std::string file_name_image("");
    switch (_image_file_name_mode) {
      case 0: {

        //ds raw KITTI file names
        char buffer_image_number[9];
        std::sprintf(buffer_image_number, "%06u", image_number);
        file_name_image = folder_images_ + "/" + buffer_image_number + ".png";
        break;
      }
      case 1: {

        //ds generate file name - SRRG version
        char buffer_image_number[9];
        std::sprintf(buffer_image_number, "%08u", image_number);
        file_name_image = folder_images_ + "/" + "camera_left.image_raw_" + buffer_image_number + ".pgm";
        break;
      }
      default: {
        throw std::runtime_error("invalid file name parsing mode");
      }
    }

    //ds check image loading and set dimensions (also checks file format)
    //ds and automatically sets parsing mode for subsequent images
    try {
      if (_number_of_image_rows == 0) {
        _initializeImageConfiguration(file_name_image);
      }
    } catch (const std::runtime_error& /*ex*/) {

      //ds generate file name - SRRG version
      char buffer_image_number[9];
      std::sprintf(buffer_image_number, "%08u", image_number);
      file_name_image = folder_images_ + "/" + "camera_left.image_raw_" + buffer_image_number + ".pgm";
      if (_number_of_image_rows == 0) {
        _initializeImageConfiguration(file_name_image);
      }

      //ds switch parsing mode (exception will not be thrown for the second image)
      _image_file_name_mode = 1;
    }

    //ds store image
    _image_poses_ground_truth[image_number] = new ImageWithPose(file_name_image, image_number, pose);
    ++image_number;
  }
  _image_poses_ground_truth.resize(image_number);
  file_ground_truth.close();
  std::cerr << "LoopClosureEvaluator::loadPosesFromFileKITTI|loaded images: " << _image_poses_ground_truth.size() << std::endl;

  //ds compute trajectory length
  for (ImageNumberQuery image_number = 1; image_number < _image_poses_ground_truth.size(); ++image_number) {

    //ds compute relative translation
    const Eigen::Isometry3d previous_to_current = _image_poses_ground_truth[image_number]->pose.inverse()*_image_poses_ground_truth[image_number-1]->pose;

    //ds update length
    _trajectory_length_meters += previous_to_current.translation().norm();
  }
  std::cerr << "LoopClosureEvaluator::loadPosesFromFileKITTI|total trajectory length (m): " << _trajectory_length_meters << std::endl;
}

void LoopClosureEvaluator::loadImagesWithPosesFromFileMalaga(const std::string& file_name_poses_ground_truth_, const std::string& images_folder_) {
  _trajectory_length_meters = 0;
  _image_poses_ground_truth.clear();

  //ds image timestamps
  std::vector<std::pair<double, std::string>> image_timestamps;

  //ds parse the image directory
  DIR* handle_directory   = 0;
  struct dirent* iterator = 0;
  if ((handle_directory = opendir(images_folder_.c_str()))) {
    while ((iterator = readdir (handle_directory))) {

      //ds buffer file name
      const std::string file_name = iterator->d_name;

      //ds check if its a left camera image (that is not a hidden linux file)
      if (file_name.find("left.jpg") != std::string::npos && file_name[0] != '.') {

        //ds look for the beginning and end of the timestamp
        const std::size_t index_begin = file_name.find("img_CAMERA1_") + std::strlen("img_CAMERA1_");
        const std::size_t index_end   = file_name.find("_left.jpg");

        //ds compute timestamp
        const double timestamp_seconds = std::stod(file_name.substr(index_begin, index_end-index_begin));

        //ds store image information
        image_timestamps.push_back(std::make_pair(timestamp_seconds, file_name));
      }
    }
    closedir(handle_directory);
  } else {
    std::cerr << "LoopClosureEvaluator::loadPosesFromFileMalaga|ERROR: unable to access image folder: " << images_folder_ << std::endl;
    throw std::runtime_error("invalid image folder");
  }

  //ds sort timestamps in ascending order
  std::sort(image_timestamps.begin(), image_timestamps.end(), [](const std::pair<double, std::string>& a_, const std::pair<double, std::string>& b_){return a_.first < b_.first;});

  //ds check failure
  if (image_timestamps.size() == 0) {
    std::cerr << "LoopClosureEvaluator::loadPosesFromFileMalaga|ERROR: unable to load images from: " << images_folder_ << std::endl;
    throw std::runtime_error("unable to load images");
  }

  //ds timestamps for pose to image synchronization
  std::vector<double> pose_timestamps;

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_poses_ground_truth_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::loadPosesFromFileMalaga|ERROR: unable to open file: " << file_name_poses_ground_truth_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds reading buffers - skip the first line (header line)
  double timestamp_seconds      = 0;
  double latitude_radians       = 0;
  double longitude_radians      = 0;
  double altitude_meters        = 0;
  uint32_t fix                  = 0;
  uint32_t number_of_satellites = 0;
  double speed_knots            = 0;
  double heading_degrees        = 0;

  std::string line_buffer("");
  std::getline(file_ground_truth, line_buffer);

  //ds start parsing all poses to the last line
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses_gps(0);
  Eigen::Isometry3d pose_gps_previous(Eigen::Isometry3d::Identity());
  const Eigen::Vector3d orientation_car(0, 0, 1);
  while (true) {

    //ds reading buffers
    line_buffer = "";
    std::getline(file_ground_truth, line_buffer);
    std::istringstream stream(line_buffer);
    if (line_buffer.empty()) {
      break;
    }

    //ds parse in fixed order
    stream >> timestamp_seconds;
    stream >> latitude_radians;
    stream >> longitude_radians;
    stream >> altitude_meters;
    stream >> fix;
    stream >> number_of_satellites;
    stream >> speed_knots;
    stream >> heading_degrees;

    //ds parse pose matrix directly from file
    Eigen::Isometry3d pose(Eigen::Isometry3d::Identity());
    for(uint8_t u = 0; u < 3; ++u) {
      stream >> pose(u,3);
    }

    //ds compute relative orientation (have nothing else)
    const Eigen::Vector3d translation_delta = pose.translation()-pose_gps_previous.translation();
    Eigen::Vector3d translation_delta_normalized(Eigen::Vector3d::Zero());
    if (translation_delta.norm() > 0) {
      translation_delta_normalized   = translation_delta/translation_delta.norm();
      Eigen::Quaterniond orientation = Eigen::Quaterniond::FromTwoVectors(orientation_car, translation_delta_normalized);
      pose.linear()                  = orientation.toRotationMatrix();

      //ds check if we can update the initial orientation
      if (poses_gps.size() == 1) {
        poses_gps.front().linear() = pose.linear();
      }
    }

    //ds set pose
    poses_gps.push_back(pose);

//      const Eigen::Vector3d difference_euler_angles(pose.linear().eulerAngles(2, 0, 2)-pose_previous.linear().eulerAngles(2, 0, 2));
//      std::cerr << "[" << timestamp_seconds << "][" << number_of_satellites << "]"
//                << " heading: " << heading_degrees
//                << " x/y/z: " << pose.translation().transpose()
//                << " euler delta: " << difference_euler_angles.transpose() << std::endl;

    pose_gps_previous = pose;
//      getchar();

    //ds keep timestamp
    pose_timestamps.push_back(timestamp_seconds);
  }
  file_ground_truth.close();
  std::cerr << "LoopClosureEvaluator::loadPosesFromFileMalaga|loaded GPS poses: " << poses_gps.size() << std::endl;

  //ds prepare final poses
  _image_poses_ground_truth.resize(image_timestamps.size());

  //ds initial pose
  pose_gps_previous = poses_gps.front();

  //ds iterators
  uint64_t image_number = 0;
  uint64_t index_pose  = 0;

  //ds until all images are processed - interpolate (assuming 1 second steps in GPS)
  while (image_number < image_timestamps.size()) {

    //ds compute current velocity
    const Eigen::Vector3d velocity = poses_gps[index_pose].translation()-pose_gps_previous.translation();

    //ds as long as the current pose timestamp is ahead of the current image
    while (image_timestamps[image_number].first <= pose_timestamps[index_pose]) {

      //ds compute timestamp delta
      const double interval_seconds = 1-(pose_timestamps[index_pose]-image_timestamps[image_number].first);
//        std::printf("%f < %f (interval: %f)\n", image_timestamps_[index_image].first, pose_timestamps[index_pose], interval_seconds);

      //ds interpolate the position based on the current delta per second (Euler integration)
      Eigen::Isometry3d pose_current(Eigen::Isometry3d::Identity());
      pose_current.translation()        = pose_gps_previous.translation()+velocity*interval_seconds;
      pose_current.linear()             = pose_gps_previous.linear();

      //ds generate file name
      const std::string file_name_image = images_folder_ + "/" + image_timestamps[image_number].second;

      //ds if image dimensions are not set yet
      if (_number_of_image_rows == 0) {
        _initializeImageConfiguration(file_name_image);
      }

      //ds store image
      _image_poses_ground_truth[image_number] = new ImageWithPose(file_name_image, image_number, pose_current);

//        std::cerr << _poses_ground_truth[index_image].translation().transpose() << " to " << poses_GPS[index_pose].translation().transpose() << std::endl;
      ++image_number;
    }

//      std::cerr << "------------------------------------" << std::endl;
//      getchar();

    //ds update previous and advance pose timestamp
    pose_gps_previous = poses_gps[index_pose];
    ++index_pose;

    //ds stop criterion
    if (index_pose == pose_timestamps.size()) {

      //ds check if some image numbers are above the gps stamps
      while (image_number < image_timestamps.size()) {

        //ds use the most recent velocity and interpolate ahead (BRUTAL)
        const double interval_seconds = 1-(pose_timestamps[index_pose-1]-image_timestamps[image_number].first);

        //ds interpolate the position based on the current delta per second (Euler integration)
        Eigen::Isometry3d pose_current(Eigen::Isometry3d::Identity());
        pose_current.translation()        = pose_gps_previous.translation()+velocity*interval_seconds;
        pose_current.linear()             = pose_gps_previous.linear();

        //ds generate file name
        const std::string file_name_image = images_folder_ + "/" + image_timestamps[image_number].second;

        //ds store image
        _image_poses_ground_truth[image_number] = new ImageWithPose(file_name_image, image_number, pose_current);

//          std::cerr << image_number << ": " << _poses_ground_truth[image_number]->pose.translation().transpose() << std::endl;
//          getchar();
        ++image_number;
      }
      break;
    }
  }

  std::cerr << "LoopClosureEvaluator::loadPosesFromFileMalaga|images: " << _image_poses_ground_truth.size()
            << " ratio per GPS pose: " << static_cast<double>(_image_poses_ground_truth.size())/poses_gps.size() << std::endl;
}

void LoopClosureEvaluator::loadImagesWithPosesFromFileLucia(const std::string& file_name_poses_ground_truth_,
                                                            const std::string& file_name_images_with_timestamps_) {
  _trajectory_length_meters = 0;
  _image_poses_ground_truth.clear();

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_poses_ground_truth_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileLucia|ERROR: unable to open file: " << file_name_poses_ground_truth_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds parse timestamped poses (GPS+IMU)
  std::vector<PoseWithTimestamp, Eigen::aligned_allocator<PoseWithTimestamp>> poses_gps;

  //ds earths radius for local path computation
  const double radius_earth_mean_meters = 6371000;
  Eigen::Vector3d initial_position(Eigen::Vector3d::Zero());

  //ds read line by line
  std::string buffer_line;
  while (std::getline(file_ground_truth, buffer_line)) {
    std::istringstream stream(buffer_line);
    if (buffer_line.empty()) {
      break;
    }

    //ds parsables
    double timestamp_seconds      = 0;
    double timestamp_milliseconds = 0;
    double latitude_degrees       = 0;
    double longitude_degrees      = 0;
    double altitude_meters        = 0;
    double height_amsl_meters     = 0;
    double velocity_x             = 0;
    double velocity_y             = 0;
    double velocity_z             = 0;
    double roll_radians           = 0;
    double pitch_radians          = 0;
    double yaw_radians            = 0;

    //ds parse in fixed order
    stream >> timestamp_seconds >> timestamp_milliseconds;
    stream >> latitude_degrees >> longitude_degrees >> altitude_meters >> height_amsl_meters;
    stream >> velocity_x >> velocity_y >> velocity_z >> roll_radians >> pitch_radians >> yaw_radians;

    //ds conversions
    const double latitude_radians  = latitude_degrees/180*M_PI;
    const double longitude_radians = longitude_degrees/180*M_PI;

    //ds if initial position is not initialized yet (start map from origin)
    if (initial_position.norm() == 0) {
      initial_position.x() = std::tan(longitude_radians)*radius_earth_mean_meters;
      initial_position.y() = std::tan(latitude_radians)*radius_earth_mean_meters;
      initial_position.z() = altitude_meters;
    }

    //ds pseudo-flat coordinates
    const double coordinate_x = std::tan(longitude_radians)*radius_earth_mean_meters-initial_position.x();
    const double coordinate_y = std::tan(latitude_radians)*radius_earth_mean_meters-initial_position.y();
    const double coordinate_z = altitude_meters-initial_position.z();

    //ds compute orientation quaternion
    Eigen::Quaterniond orientation;
    orientation = Eigen::AngleAxisd(roll_radians, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(pitch_radians, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(yaw_radians, Eigen::Vector3d::UnitZ());

    //ds compose isometry
    Eigen::Isometry3d pose(Eigen::Isometry3d::Identity());
    pose.translation().x() = coordinate_x;
    pose.translation().y() = coordinate_y;
    pose.translation().z() = coordinate_z;

    //ds rotate into camera frame
    Eigen::Matrix3d orientation_robot_to_camera(Eigen::Matrix3d::Identity());
    orientation_robot_to_camera << 0, 0, 1,
                                   0, 1, 0,
                                   -1, 0, 0;
    pose.linear() = orientation_robot_to_camera*orientation.toRotationMatrix();

    //ds compute full timestamp
    const double full_timestamp_seconds = timestamp_seconds + timestamp_milliseconds/1e6;

    //ds add pose
    poses_gps.push_back(PoseWithTimestamp(pose, full_timestamp_seconds));
  }
  file_ground_truth.close();
  std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileLucia|ground truth poses: " << poses_gps.size() << std::endl;

  //ds load ground truth text file
  std::ifstream file_images(file_name_images_with_timestamps_, std::ifstream::in);

  //ds check failure
  if (!file_images.is_open() || !file_images.good()) {
    std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileLucia|ERROR: unable to open file: " << file_name_images_with_timestamps_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds iterate over image file log
  while (std::getline(file_images, buffer_line)) {

    //ds skip the line if it does not contain any image-related information
    if (buffer_line.find(".bmp") == std::string::npos) {
      continue;
    }

    //ds get line to a parsing buffer
    std::istringstream stream(buffer_line);
    if (buffer_line.empty()) {
      break;
    }

    //ds parsables
    double timestamp_seconds_cam0      = 0;
    double timestamp_milliseconds_cam0 = 0;
    uint32_t image_width_pixels        = 0;
    uint32_t image_height_pixels       = 0;
    std::string image_format           = "";
    double timestamp_seconds_cam1      = 0;
    double timestamp_milliseconds_cam1 = 0;
    std::string image_filename_cam0    = "";
    std::string image_filename_cam1    = "";

    //ds parse in fixed order
    stream >> timestamp_seconds_cam0 >> timestamp_milliseconds_cam0;
    stream >> image_width_pixels >> image_height_pixels >> image_format;
    stream >> timestamp_seconds_cam1 >> timestamp_milliseconds_cam1;
    stream >> image_width_pixels >> image_height_pixels >> image_format;
    stream >> image_filename_cam0 >> image_filename_cam1;

    //ds compute full timestamp
    const double full_timestamp_seconds = timestamp_seconds_cam0 + timestamp_milliseconds_cam0/1e6;

    //ds look for a matching pose in the poses buffer (brute force, assuming ordered by timestamps)
    Eigen::Isometry3d pose_best(Eigen::Isometry3d::Identity());
    double time_delta_best = 1e9;
    for (const PoseWithTimestamp& pose: poses_gps) {

      //ds compute current time delta
      const double time_delta_seconds = std::fabs(pose.timestamp_seconds-full_timestamp_seconds);

      //ds if better than the previous, update previous and continue the search
      if (time_delta_seconds < time_delta_best) {
        time_delta_best = time_delta_seconds;
        pose_best       = pose.pose;
      } else {
        break;
      }
    }

    //ds if we found a matching measurement
    if (time_delta_best < 1) {

      //ds if image dimensions are not set yet
      if (_number_of_image_rows == 0) {
        _initializeImageConfiguration(image_filename_cam0, true);
      }

      //ds add image with matching pose to bookkeeping
      _image_poses_ground_truth.push_back(new ImageWithPose(image_filename_cam0, _image_poses_ground_truth.size(), pose_best));
    }
  }
  file_images.close();
  std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileLucia|images: " << _image_poses_ground_truth.size()
            << " ratio per GPS pose: " << static_cast<double>(_image_poses_ground_truth.size())/poses_gps.size() << std::endl;
}

void LoopClosureEvaluator::loadImagesWithPosesFromFileOxford(const std::string& file_name_poses_ground_truth_0_,
                                                             const std::string& images_folder_0_,
                                                             const std::string& file_name_poses_ground_truth_1_,
                                                             const std::string& images_folder_1_) {
  _trajectory_length_meters = 0;
  _image_poses_ground_truth.clear();

  //ds obtain pose data for both sequences
  PoseWithTimestampVector poses_0(_getPosesFromGPSOxford(file_name_poses_ground_truth_0_));
  PoseWithTimestampVector poses_1(_getPosesFromGPSOxford(file_name_poses_ground_truth_1_));

  //ds merge pose information
  PoseWithTimestampVector poses(poses_0);
  poses.insert(poses.end(), poses_1.begin(), poses_1.end());
  std::sort(poses.begin(), poses.end(), [](const PoseWithTimestamp& a_, const PoseWithTimestamp& b_){return a_.timestamp_seconds < b_.timestamp_seconds;});

  //ds load images for both sequences
  ImageFileWithTimestampVector images_files_0(_getImageFilesFromGPSOxford(images_folder_0_));
  ImageFileWithTimestampVector images_files_1(_getImageFilesFromGPSOxford(images_folder_1_));

  //ds merge image information
  ImageFileWithTimestampVector image_files(images_files_0);
  image_files.insert(image_files.end(), images_files_1.begin(), images_files_1.end());
  std::sort(image_files.begin(), image_files.end(), [](const ImageFileWithTimestamp& a_, const ImageFileWithTimestamp& b_){return a_.timestamp_seconds < b_.timestamp_seconds;});

  //ds for all images
  for (const ImageFileWithTimestamp& image_file: image_files) {

    //ds look for a matching pose in the poses buffer (brute force, assuming ordered by timestamps)
    Eigen::Isometry3d pose_best(Eigen::Isometry3d::Identity());
    double time_delta_best = 1e9;
    for (const PoseWithTimestamp& pose: poses) {

      //ds compute current time delta
      const double time_delta_seconds = std::fabs(pose.timestamp_seconds-image_file.timestamp_seconds);

      //ds if better than the previous, update previous and continue the search
      if (time_delta_seconds < time_delta_best) {
        time_delta_best = time_delta_seconds;
        pose_best       = pose.pose;
      } else {
        break;
      }
    }

    //ds if we found a matching measurement
    if (time_delta_best < 1) {

      //ds if image dimensions are not set yet
      if (_number_of_image_rows == 0) {
        _initializeImageConfiguration(image_file.file_name, true);
      }

      //ds add image_file with matching pose to bookkeeping
      _image_poses_ground_truth.push_back(new ImageWithPose(image_file.file_name, _image_poses_ground_truth.size(), pose_best));
    }
  }

  std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileOxford|images: " << _image_poses_ground_truth.size()
            << " ratio per odometry pose: " << static_cast<double>(_image_poses_ground_truth.size())/poses.size() << std::endl;
}

void LoopClosureEvaluator::loadImagesWithPosesFromFileNordland(const std::string& file_name_poses_ground_truth_query_,
                                                               const std::string& images_folder_query_,
                                                               const std::string& file_name_poses_ground_reference_,
                                                               const std::string& images_folder_reference_) {
  _trajectory_length_meters = 0;
  _image_poses_ground_truth.clear();

  //ds obtain pose data for both sequences
  PoseWithTimestampVector poses_query(_getPosesFromGPSNordland(file_name_poses_ground_truth_query_));
  PoseWithTimestampVector poses_reference(_getPosesFromGPSNordland(file_name_poses_ground_reference_));

  Eigen::Isometry3d pose_previous(poses_reference.front().pose);
  for (const PoseWithTimestamp& pose: poses_reference) {
    const double translation_per_image = (pose.pose.inverse()*pose_previous).translation().norm();

//    //ds compute average translation to filter bad measurements
//    if (_trajectory_length_meters > 100) {
//      const double average_translation_per_image = static_cast<double>(_trajectory_length_meters)/_image_poses_ground_truth.size();
//      if (translation_per_image/average_translation_per_image > 5) {
//        pose_previous = pose.pose;
//        continue;
//      }
//    }

    //ds add measurement
    _trajectory_length_meters += translation_per_image;
    _image_poses_ground_truth.push_back(new ImageWithPose(std::to_string(_image_poses_ground_truth.size()), _image_poses_ground_truth.size(), pose.pose));
    pose_previous = pose.pose;
  }

  std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileNordland|images: " << _image_poses_ground_truth.size()
            << " ratio per odometry pose: " << static_cast<double>(_image_poses_ground_truth.size())/poses_reference.size() << std::endl;
  std::cerr << "LoopClosureEvaluator::loadImagesWithPosesFromFileNordland|total trajectory length (m): " << _trajectory_length_meters << std::endl;
}

void LoopClosureEvaluator::loadImagesFromDirectoryZubud(const std::string& directory_query_, const std::string& directory_reference_) {
  _image_poses_ground_truth.clear();
  _image_poses_query.clear();

  //ds load query and reference image set
  std::vector<std::string> image_paths_query(0);
  std::vector<std::string> image_paths_reference(0);
  _loadImagePathsFromDirectory(directory_query_, "JPG", image_paths_query);
  _loadImagePathsFromDirectory(directory_reference_, "png", image_paths_reference);

  //ds store query images in query vector
  for (const std::string& image_path: image_paths_query) {
    const size_t index_start = image_path.find("qimg")+4;
    const size_t index_stop  = image_path.find(".JPG", index_start);

    //ds store image with default pose
    ImageWithPose* image = new ImageWithPose(image_path,
                                             std::stoi(image_path.substr(index_start, index_stop-index_start)),
                                             Eigen::Isometry3d::Identity());
    _image_poses_query.push_back(image);
  }

  //ds store reference images in ground truth
  for (const std::string& image_path: image_paths_reference) {
    const size_t index_start = image_path.find("object")+6;
    const size_t index_stop  = image_path.find(".view", index_start);

    //ds store image with default pose
    ImageWithPose* image = new ImageWithPose(image_path,
                                             std::stoi(image_path.substr(index_start, index_stop-index_start)),
                                             Eigen::Isometry3d::Identity());
    _image_poses_ground_truth.push_back(image);
  }

  std::cerr << "LoopClosureEvaluator::loadImagesFromDirectoryZubud|loaded query images: " << _image_poses_query.size() << std::endl;
  std::cerr << "LoopClosureEvaluator::loadImagesFromDirectoryZubud|loaded reference images: " << _image_poses_ground_truth.size() << std::endl;
}

void LoopClosureEvaluator::computeLoopClosureFeasibilityMap(const uint32_t& image_number_start_,
                                                            const uint32_t& image_number_stop_,
                                                            const uint32_t& interspace_image_number_,
                                                            const double& maximum_difference_position_meters_,
                                                            const double& maximum_difference_angle_radians_,
                                                            const uint32_t& minimum_distance_between_closure_images_) {

  //ds determine available subset of images
  uint32_t image_number_stop = _image_poses_ground_truth.size()-1;
  if (image_number_stop_ != 0) {
    image_number_stop = image_number_stop_;
  }

  //ds compute the total number of queries we can receive
  const uint64_t number_of_database_queries = _image_poses_ground_truth.size()/interspace_image_number_;
  _closure_feasability_map.clear();
  _valid_query_image_numbers.clear();
  _valid_train_image_numbers.clear();
  _total_number_of_valid_closures = 0;

  //ds loop over the buffered poses in reverse order (since we cannot close loops forwards) - stopping as we cannot close further
  for (uint32_t image_number_query = image_number_stop; image_number_query >= std::max(image_number_start_, minimum_distance_between_closure_images_); --image_number_query) {

    //ds if we got a query image
    if (image_number_query%interspace_image_number_ == 0) {

      //ds initialize closures found for this query image
      std::multiset<ImageNumberTrain> closed_training_images;

      //ds available closure range (must be above the minimum threshold, to avoid reporting closures when the vehicle was standing still)
      const ImageNumberQuery image_number_last = std::max(image_number_query-minimum_distance_between_closure_images_+1, image_number_start_);

      //ds loop over closure candidates
      for (ImageNumberTrain image_number_train = image_number_start_; image_number_train < image_number_last; ++image_number_train) {

        //ds if we got a query image
        if (image_number_train%interspace_image_number_ == 0) {

          //ds readability
          const Eigen::Isometry3d& pose_query     = _image_poses_ground_truth[image_number_query]->pose;
          const Eigen::Isometry3d& pose_reference = _image_poses_ground_truth[image_number_train]->pose;

          //ds check closure criteria I - position vicinity
          if ((pose_query.translation()-pose_reference.translation()).norm() < maximum_difference_position_meters_) {

            //ds check closure critera II - orientation similarity
            const Eigen::Quaterniond quaternion_query(pose_query.linear());
            const Eigen::Quaterniond quaternion_reference(pose_reference.linear());
            if (quaternion_query.angularDistance(quaternion_reference) < maximum_difference_angle_radians_) {

              //ds bookkeeping
              _valid_query_image_numbers.insert(_image_poses_ground_truth[image_number_query]);
              _valid_train_image_numbers.insert(_image_poses_ground_truth[image_number_train]);
              ++_total_number_of_valid_closures;

              //ds add the closure mapping
              closed_training_images.insert(image_number_train);
            }
          }
        }
      }

      //ds if closures were found
      if (closed_training_images.size() > 0) {
        _closure_feasability_map.insert(std::make_pair(image_number_query, closed_training_images));
      }
    }
  }
  std::cerr << "LoopClosureEvaluator::computeLoopClosureFeasabilityMap|computed feasible total number of closures: " << _total_number_of_valid_closures
            << " (average per image: " << static_cast<double>(_total_number_of_valid_closures)/number_of_database_queries << ")" << std::endl;
}

void LoopClosureEvaluator::computeLoopClosureFeasibilityMap(const std::string& file_name_ground_truth_mapping_, const char& separator_) {

  //ds clear
  _closure_feasability_map.clear();
  _valid_query_image_numbers.clear();
  _valid_train_image_numbers.clear();
  _total_number_of_valid_closures = 0;

  //ds parse QUERY to REFERENCE number mapping
  std::ifstream file_ground_truth_mapping(file_name_ground_truth_mapping_);
  std::string buffer_line;
  while (std::getline(file_ground_truth_mapping, buffer_line)) {

    //ds parse query image number
    size_t index  = buffer_line.find_first_of(separator_);
    if (index == std::string::npos) {
      continue;
    }

    //ds candidates
    ImageNumberQuery image_number_query     = 0;
    ImageNumberTrain image_number_reference = 0;

    //ds convert query image number
    try {
      image_number_query = std::stoi(buffer_line.substr(0, index));
    } catch (const std::invalid_argument& /*ex*/) {
      continue;
    }

    //ds parse reference image number
    index = buffer_line.find_first_not_of(' ', index+1);
    if (index == std::string::npos) {
      continue;
    }

    //ds convert reference image number
    try {
      image_number_reference = std::stoi(buffer_line.substr(index));
    } catch (const std::invalid_argument& /*ex*/) {
      continue;
    }

    //ds closure
    const ImageWithPose* image_query = 0;

    //ds look for the query image with this number
    for (const ImageWithPose* image: _image_poses_query) {
      if (image->image_number == image_number_query) {
        image_query = image;
        break;
      }
    }
    _valid_query_image_numbers.insert(image_query);

    //ds look for all reference images with this number (yeiks)
    std::multiset<ImageNumberTrain> closed_reference_images;
    for (const ImageWithPose* image: _image_poses_ground_truth) {
      if (image->image_number == image_number_reference) {

        //ds register closure
        _valid_train_image_numbers.insert(image);
        closed_reference_images.insert(image->image_number);
        ++_total_number_of_valid_closures;
      }
    }

    //ds if closures were found
    if (closed_reference_images.size() > 0) {
      _closure_feasability_map.insert(std::make_pair(image_number_query, closed_reference_images));
    }
  }

  std::cerr << "LoopClosureEvaluator::computeLoopClosureFeasibilityMap|computed feasible total number of closures: " << _total_number_of_valid_closures << std::endl;
}

std::pair<double, double> LoopClosureEvaluator::getPrecisionRecall(ImagePairVector& reported_closures_, const double& target_recall_) {
  if (!_reached_target_display_recall) {
    _valid_closures.clear();
    _invalid_closures.clear();
  }

  //ds check if the ground truth is not computed
  if (_closure_feasability_map.empty()) {
    std::cerr << "LoopClosureEvaluator::getPrecisionRecall|ERROR: ground truth closure feasibility map not computed - call computeLoopClosureFeasabilityMap first" << std::endl;
    throw std::runtime_error("LoopClosureEvaluator::getPrecisionRecall|ERROR: ground truth closure feasibility map not computed - call computeLoopClosureFeasabilityMap first");
  }

//    std::cerr << "LoopClosureEvaluator::getPrecisionRecall|computing Precision/Recall for closure candidates: " << reported_closures_.size() << std::endl;

  //ds counters
  uint32_t number_of_reported_closures           = 0;
  uint32_t number_of_correctly_reported_closures = 0;
  uint32_t total_number_of_valid_closures        = target_recall_*_total_number_of_valid_closures;

  //ds adjust target closure number if a BF filtered ground truth is available
  if (_closure_map_bf.size() > 0) {

    //ds compute correct closures in reported pool
    for (ImageNumberAssociation& reported_closure: reported_closures_) {
      reported_closure.valid = false;
      ++number_of_reported_closures;

      //ds if there is at least one reference image available (the image is closable)
      if (_closure_map_bf.find(reported_closure.query) != _closure_map_bf.end()) {

        //ds check if the reference image is in the list
        if (_closure_map_bf.at(reported_closure.query).count(reported_closure.train)) {
          ++number_of_correctly_reported_closures;
          reported_closure.valid = true;
        }
      }

      //ds visualization (show additional, correctly detected closures which were not obtained in the BF ground truth)
      if (!_reached_target_display_recall) {
        _valid_closures.push_back(std::make_pair(_image_poses_ground_truth.at(reported_closure.query),
                                                 _image_poses_ground_truth.at(reported_closure.train)));
      }

      //ds check if maximum recall is reached
      if (number_of_correctly_reported_closures == total_number_of_valid_closures) {
        break;
      }
    }
  } else {

    //ds compute correct closures in reported pool
    for (ImageNumberAssociation& reported_closure: reported_closures_) {
      ++number_of_reported_closures;

      //ds if there is at least one reference image available (the image is closable)
      if (_closure_feasability_map.find(reported_closure.query) != _closure_feasability_map.end()) {

        //ds check if the reference image is in the list
        if (_closure_feasability_map.at(reported_closure.query).count(reported_closure.train)) {
          ++number_of_correctly_reported_closures;
          reported_closure.valid = true;
        }
      }

      //ds if valid
      if (!_reached_target_display_recall) {
        if (reported_closure.valid) {
          _valid_closures.push_back(std::make_pair(_image_poses_ground_truth.at(reported_closure.query),
                                                   _image_poses_ground_truth.at(reported_closure.train)));
        } else {
          _invalid_closures.push_back(std::make_pair(_image_poses_ground_truth.at(reported_closure.query),
                                                     _image_poses_ground_truth.at(reported_closure.train)));
        }
      }

      //ds check if maximum recall is reached
      if (number_of_correctly_reported_closures == total_number_of_valid_closures) {
        break;
      }
    }
  }

  //ds precision: correctly reported/reported closures
  double precision = static_cast<double>(number_of_correctly_reported_closures)/number_of_reported_closures;

  //ds recall: correct/possible closures
  double recall = static_cast<double>(number_of_correctly_reported_closures)/total_number_of_valid_closures;

  //ds done
//    std::cerr << "LoopClosureEvaluator::getPrecisionRecall|Precision                (CORRECT/REPORTED): " << precision << " (" << number_of_correctly_reported_closures << "/" << number_of_reported_closures << ")" << std::endl;
//    std::cerr << "LoopClosureEvaluator::getPrecisionRecall|Recall (CORRECT_SINGULAR/POSSIBLE_SINGULAR): " << recall << " (" << closed_images_reported_singular.size() << "/" << number_of_possible_closures_singular << ")" << std::endl;
  return std::make_pair(precision, recall);
}

std::vector<std::pair<double, double>> LoopClosureEvaluator::computePrecisionRecallCurve(std::vector<ResultImageRetrieval>& reported_closures_,
                                                                                         double& maximum_f1_score_,
                                                                                         const double& target_recall_,
                                                                                         const std::string& file_name_) {
  _reached_target_display_recall = false;

  //ds check if the ground truth is not computed
  if (_closure_feasability_map.empty()) {
    std::cerr << "LoopClosureEvaluator::computePrecisionRecallCurve|ERROR: ground truth closure feasibility map not computed - call computeLoopClosureFeasabilityMap first" << std::endl;
    throw std::runtime_error("LoopClosureEvaluator::computePrecisionRecallCurve|ERROR: ground truth closure feasibility map not computed - call computeLoopClosureFeasabilityMap first");
  }

  //ds sort input vector in descending order by score (start evaluation at maximum precision and minimum recall)
  std::sort(reported_closures_.begin(), reported_closures_.end(), [](const ResultImageRetrieval& a_, const ResultImageRetrieval& b_){return (a_.number_of_matches_relative > b_.number_of_matches_relative);});

  //ds output vector with precision recall values
  std::vector<std::pair<double, double>> precision_recall_values(0);

  //ds if dumping is desired
  std::ofstream outfile_precision_recall;
  if (!file_name_.empty()) {

    //ds precision/recall stats to save to file
    outfile_precision_recall.open(file_name_, std::ifstream::out);

    //ds header
    std::printf("==========================================\n");
    std::printf("PERCENTAGE | REPORTED | PRECISION | RECALL\n");

    //ds write file header
    outfile_precision_recall << "#PERCENTAGE REPORTED PRECISION RECALL\n";
    outfile_precision_recall << 0 << " " << 0 << " " << 1 << " " << 0 << "\n";
  }
  precision_recall_values.push_back(std::make_pair(0, 1));

  //ds best F1 score
  maximum_f1_score_ = 0;

  //ds check 10s percentiles starting from top (highest matching score)
  const double multiplier             = 1.2;
  const double resolution             = 10000.0;
  double percentile                   = 1;
  const uint32_t number_of_iterations = 100;
  for (uint32_t iteration = 1; iteration < number_of_iterations; ++iteration) {

    //ds compute current percentage
    double percentage = percentile/resolution;

    //ds check limit
    if (percentage >= 1) {

      //ds make this the last round at 100 percent
      percentage = 1;
      iteration  = number_of_iterations;
    }

    //ds compute target number of top closures to add
    const ImagePairVector::size_type number_of_closures_in_percentile = std::round(percentage*reported_closures_.size());

    //ds if we have closures in the percentile
    if (number_of_closures_in_percentile > 0) {

      //ds closures to check in current percentile
      ImagePairVector reported_closures_percentile(number_of_closures_in_percentile);

      //ds add closures to comparing vector
      for (uint32_t index_closure = 0; index_closure < number_of_closures_in_percentile; ++index_closure) {
        reported_closures_percentile[index_closure] = reported_closures_[index_closure].image_association;
      }

      //ds obtain precision/recall values
      std::pair<double, double> precision_recall = getPrecisionRecall(reported_closures_percentile, target_recall_);
      if (outfile_precision_recall.is_open()) {
        std::printf("   %7.5f |  %7lu |   %6.4f  | %6.4f\n", percentage, reported_closures_percentile.size(), precision_recall.first, precision_recall.second);
      }
      precision_recall_values.push_back(precision_recall);

      //ds check minimum recall for visualization
      if (precision_recall.second > 0.7) {
        _reached_target_display_recall = true;
      }

      //ds update closure validity in out vector
      for (uint32_t index_closure = 0; index_closure < number_of_closures_in_percentile; ++index_closure) {
        reported_closures_[index_closure].image_association.valid = reported_closures_percentile[index_closure].valid;
      }

      //ds compute f1 score
      const double f1_score = 2*(precision_recall.first*precision_recall.second)/(precision_recall.first+precision_recall.second);
      if (f1_score > maximum_f1_score_) {
        maximum_f1_score_ = f1_score;
      }

      //ds stream to file
      if (outfile_precision_recall.is_open()) {
        outfile_precision_recall << percentage << " " << reported_closures_percentile.size() << " " << precision_recall.first << " " << precision_recall.second << "\n";
      }

      //ds check if we achieved full recall
      if (precision_recall.second == 1) {
        break;
      }
    }

    //ds update percentile
    percentile *= multiplier;
  }

  //ds close stream
  if (outfile_precision_recall.is_open()) {
    std::printf("==========================================\n");
    std::printf("maximum F1 score: %f\n", maximum_f1_score_);
    std::printf("==========================================\n");
    outfile_precision_recall.close();
  }
  return precision_recall_values;
}

void LoopClosureEvaluator::loadClosures(const std::string& file_name_closures_,
                                        const uint32_t& image_number_start_,
                                        const uint32_t& image_number_stop_,
                                        const uint32_t& interspace_image_number_,
                                        const double& maximum_difference_position_meters_,
                                        const double& maximum_difference_angle_radians_,
                                        const int32_t& minimum_distance_between_closure_images_) {

  //ds compute full feasibility (in order to not reject correct but infeasible Brute-force closures)
  computeLoopClosureFeasibilityMap(image_number_start_,
                                   image_number_stop_,
                                   interspace_image_number_,
                                   maximum_difference_position_meters_,
                                   maximum_difference_angle_radians_,
                                   minimum_distance_between_closure_images_);

  //ds prepare filtered feasibility
  _closure_map_bf.clear();
  std::set<ImageNumberQuery> query_image_numbers;
  std::set<ImageNumberTrain> train_image_numbers;
  _total_number_of_valid_closures = 0;

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_closures_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::loadClosures|ERROR: unable to open file: " << file_name_closures_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds read line by line
  std::string buffer_line;
  while (std::getline(file_ground_truth, buffer_line)) {

    //ds get line to a parsing buffer
    std::istringstream stream(buffer_line);
    if (buffer_line.empty()) {
      break;
    }

    //ds if comment skip parsing
    if (buffer_line.find("#") != std::string::npos) {
      continue;
    }

    //ds data fields: query - train
    ImageNumberQuery query = 0;
    ImageNumberTrain train = 0;

    //ds parse in fixed order
    stream >> query >> train;

    //ds bookkeeping
    query_image_numbers.insert(query);
    train_image_numbers.insert(train);
    ++_total_number_of_valid_closures;

    //ds check if we have no entry yet for this query
    if (_closure_map_bf.find(query) == _closure_map_bf.end()) {

      //ds insert a new entry
      std::multiset<ImageNumberTrain> train_image_numbers;
      train_image_numbers.insert(train);
      _closure_map_bf.insert(std::make_pair(query, train_image_numbers));
    } else {

      //ds update the existing entry with another valid train image number
      _closure_map_bf.at(query).insert(train);
    }
  }
  file_ground_truth.close();
  std::cerr << "LoopClosureEvaluator::loadClosures|loaded closures: " << _total_number_of_valid_closures
            << " (query: " << query_image_numbers.size()
            << ", train: " << train_image_numbers.size() << ")" << std::endl;
}

PoseWithTimestampVector LoopClosureEvaluator::_getPosesFromGPSOxford(const std::string& file_name_poses_ground_truth_) const {
  if (file_name_poses_ground_truth_ == "") {
    return PoseWithTimestampVector();
  }

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_poses_ground_truth_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::_getPosesFromGPSOxford|ERROR: unable to open file: " << file_name_poses_ground_truth_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds parse timestamped poses (visual odometry)
  PoseWithTimestampVector poses;

  //ds earths radius for local path computation
  const double radius_earth_mean_meters = 6371000;
  Eigen::Vector3d initial_position(Eigen::Vector3d::Zero());

  //ds read line by line
  std::string buffer_line;
  while (std::getline(file_ground_truth, buffer_line)) {

    //ds replace all commas with spaces for streamed parsing
    std::replace(buffer_line.begin(), buffer_line.end(), ',', ' ');
    std::istringstream stream(buffer_line);

    //ds if line contains text/comments - skip it
    if (buffer_line.find("timestamp") != std::string::npos) {
      continue;
    }

    //ds if line is empty - terminate
    if (buffer_line.empty()) {
      break;
    }

    //ds parsables
    double timestamp_seconds  = 0;
    std::string ins_status    = "";
    double latitude_degrees   = 0;
    double longitude_degrees  = 0;
    double altitude_meters    = 0;
    double northing           = 0;
    double easting            = 0;
    double down               = 0;
    std::string time_zone     = "";
    double velocity_north     = 0;
    double velocity_east      = 0;
    double velocity_down      = 0;
    double roll_radians       = 0;
    double pitch_radians      = 0;
    double yaw_radians        = 0;

    //ds parse in fixed order
    stream >> timestamp_seconds >> ins_status;
    stream >> latitude_degrees >> longitude_degrees >> altitude_meters;
    stream >> northing >> easting >> down;
    stream >> time_zone;
    stream >> velocity_north >> velocity_east >> velocity_down;
    stream >> roll_radians >> pitch_radians >> yaw_radians;

    //ds conversions
    const double latitude_radians  = latitude_degrees/180*M_PI;
    const double longitude_radians = longitude_degrees/180*M_PI;
    timestamp_seconds /= 1e6;

    //ds if initial position is not initialized yet (start map from origin)
    if (initial_position.norm() == 0) {
      initial_position.x() = std::tan(longitude_radians)*radius_earth_mean_meters;
      initial_position.y() = std::tan(latitude_radians)*radius_earth_mean_meters;
      initial_position.z() = altitude_meters;
    }

    //ds pseudo-flat coordinates
    const double coordinate_x = std::tan(longitude_radians)*radius_earth_mean_meters-initial_position.x();
    const double coordinate_y = std::tan(latitude_radians)*radius_earth_mean_meters-initial_position.y();
    const double coordinate_z = altitude_meters-initial_position.z();

    //ds compute orientation quaternion
    Eigen::Quaterniond orientation;
    orientation = Eigen::AngleAxisd(roll_radians, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(pitch_radians, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(yaw_radians, Eigen::Vector3d::UnitZ());

    //ds compose isometry
    Eigen::Isometry3d pose(Eigen::Isometry3d::Identity());
    pose.translation().x() = coordinate_x;
    pose.translation().y() = coordinate_y;
    pose.translation().z() = coordinate_z;

    //ds rotate into camera frame
    Eigen::Matrix3d orientation_robot_to_camera(Eigen::Matrix3d::Identity());
    orientation_robot_to_camera << 0, 0, 1,
                                   -1, 0, 0,
                                   0, -1, 0;
    pose.linear() = orientation_robot_to_camera*orientation.toRotationMatrix();

    //ds add pose
    poses.push_back(PoseWithTimestamp(pose, timestamp_seconds));
  }
  file_ground_truth.close();

  //ds check if we failed to parse poses
  if (poses.size() == 0) {
    std::cerr << "LoopClosureEvaluator::_getPosesFromGPSOxford|ERROR: unable to parse poses from: " << file_name_poses_ground_truth_ << std::endl;
    throw std::runtime_error("unable to parse poses");
  }

  std::cerr << "LoopClosureEvaluator::_getPosesFromGPSOxford|ground truth poses: " << poses.size() << " (" << file_name_poses_ground_truth_ << ")" << std::endl;
  return poses;
}

ImageFileWithTimestampVector LoopClosureEvaluator::_getImageFilesFromGPSOxford(const std::string& folder_images_) const {
  if (folder_images_ == "") {
    return ImageFileWithTimestampVector();
  }

  //ds image timestamps (we need to sort them first with ascending timestamps)
  ImageFileWithTimestampVector image_files;

  //ds parse the image directory
  DIR* handle_directory   = 0;
  struct dirent* iterator = 0;
  if ((handle_directory = opendir(folder_images_.c_str()))) {
    while ((iterator = readdir(handle_directory))) {

      //ds buffer file name
      const std::string file_name = iterator->d_name;

      //ds check if we got an image file at hand (that is not a hidden linux file)
      if (file_name.find(".png") != std::string::npos && file_name[0] != '.') {

        //ds generate full file name
        const std::string file_name_image = folder_images_ + "/" + file_name;

        //ds look for the beginning and end of the timestamp
        const std::size_t index_end = file_name.find(".png");

        //ds compute timestamp
        const double timestamp_seconds = std::stod(file_name.substr(0, index_end))/1e6;

        //ds store image information
        image_files.push_back(ImageFileWithTimestamp(file_name_image, timestamp_seconds));
      }
    }
    closedir(handle_directory);
  } else {
    std::cerr << "LoopClosureEvaluator::_getImagesFromGPSOxford|ERROR: unable to access image folder: " << folder_images_ << std::endl;
    throw std::runtime_error("invalid image folder");
  }

  //ds check failure
  if (image_files.size() == 0) {
    std::cerr << "LoopClosureEvaluator::_getImagesFromGPSOxford|ERROR: unable to load images from: " << folder_images_ << std::endl;
    throw std::runtime_error("unable to load images");
  }

  std::cerr << "LoopClosureEvaluator::_getImagesFromGPSOxford|images: " << image_files.size() << " (" << folder_images_ << ")" << std::endl;
  return image_files;
}

void LoopClosureEvaluator::_initializeImageConfiguration(const std::string& image_file_name_, const bool& bayer_decoding_) {

  //ds load image from disk
  cv::Mat image = cv::imread(image_file_name_, CV_LOAD_IMAGE_GRAYSCALE);

  //ds if the image_file is invalid
  if (image.rows == 0 || image.cols == 0) {
    std::cerr << "LoopClosureEvaluator::_initializeImageConfiguration|ERROR: invalid image found at: " << image_file_name_ << std::endl;
    throw std::runtime_error("invalid images provided");
  }

  //ds prompt user for inspection (with optional bayer decoding)
  if (bayer_decoding_) {
    cv::cvtColor(image, image, CV_BayerGR2GRAY);
  }
  std::cerr << "LoopClosureEvaluator::_initializeImageConfiguration|loaded initial image for inspection, press any key to continue" << std::endl;
  cv::imshow(image_file_name_, image);
  cv::waitKey(0);
  cv::destroyWindow(image_file_name_);

  //ds set dimensions
  _number_of_image_rows = image.rows;
  _number_of_image_cols = image.cols;
}

PoseWithTimestampVector LoopClosureEvaluator::_getPosesFromGPSNordland(const std::string& file_name_poses_) const {
  if (file_name_poses_ == "") {
    return PoseWithTimestampVector();
  }

  //ds load ground truth text file
  std::ifstream file_ground_truth(file_name_poses_, std::ifstream::in);

  //ds check failure
  if (!file_ground_truth.is_open() || !file_ground_truth.good()) {
    std::cerr << "LoopClosureEvaluator::_getPosesFromGPSNordland|ERROR: unable to open file: " << file_name_poses_ << std::endl;
    throw std::runtime_error("unable to open file");
  }

  //ds parse timestamped poses (visual odometry)
  PoseWithTimestampVector poses;

  //ds earths radius for local path computation
  const double radius_earth_mean_meters = 6371000;
  Eigen::Vector3d initial_position(Eigen::Vector3d::Zero());

  //ds read line by line (nordlandsbanen csv uses carriage returns before newlines)
  std::string buffer_line;
  while (std::getline(file_ground_truth, buffer_line, '\r')) {

    //ds replace all commas with spaces for streamed parsing
    std::replace(buffer_line.begin(), buffer_line.end(), ',', ' ');
    std::istringstream stream(buffer_line);

    //ds if line contains text/comments - skip it
    if (buffer_line.find("tid") != std::string::npos) {
      continue;
    }

    //ds if line is empty - terminate
    if (buffer_line.empty()) {
      break;
    }

    //ds parsables
    double tid    = 0;
    double lat    = 0;
    double lon    = 0;
    double speed  = 0;
    double course = 0;
    double alt    = 0;

    //ds parse in fixed order
    stream >> tid >> lat >> lon >> speed >> course >> alt;
    const double latitude_degrees  = lat/100000;
    const double longitude_degrees = lon/100000;
    const double altitude_meters   = alt/100;

    //ds conversions
    const double latitude_radians  = latitude_degrees/180*M_PI;
    const double longitude_radians = longitude_degrees/180*M_PI;

    //ds if initial position is not initialized yet (start map from origin)
    if (initial_position.norm() == 0) {
      initial_position.x() = std::tan(longitude_radians)*radius_earth_mean_meters;
      initial_position.y() = std::tan(latitude_radians)*radius_earth_mean_meters;
      initial_position.z() = altitude_meters;
    }

    //ds pseudo-flat coordinates
    const double coordinate_x = std::tan(longitude_radians)*radius_earth_mean_meters-initial_position.x();
    const double coordinate_y = std::tan(latitude_radians)*radius_earth_mean_meters-initial_position.y();
    const double coordinate_z = altitude_meters-initial_position.z();

    //ds compose isometry
    Eigen::Isometry3d pose(Eigen::Isometry3d::Identity());
    pose.translation().x() = coordinate_x;
    pose.translation().y() = coordinate_y;
    pose.translation().z() = coordinate_z;

    //ds rotate into camera frame
    Eigen::Matrix3d orientation_robot_to_camera(Eigen::Matrix3d::Identity());
    orientation_robot_to_camera << 0, 0, 1,
                                   -1, 0, 0,
                                   0, -1, 0;
    pose.linear() = orientation_robot_to_camera;

    //ds add pose
    poses.push_back(PoseWithTimestamp(pose, tid));
  }
  file_ground_truth.close();

  //ds check if we failed to parse poses
  if (poses.size() == 0) {
    std::cerr << "LoopClosureEvaluator::_getPosesFromGPSNordland|ERROR: unable to parse poses from: " << file_name_poses_ << std::endl;
    throw std::runtime_error("unable to parse poses");
  }

  std::cerr << "LoopClosureEvaluator::_getPosesFromGPSNordland|ground truth poses: " << poses.size() << " (" << file_name_poses_ << ")" << std::endl;
  return poses;
}

void LoopClosureEvaluator::_loadImagePathsFromDirectory(const std::string& directory_,
                                                        const std::string& image_file_name_,
                                                        std::vector<std::string>& image_paths_) const {

  //ds parse the image directory
  DIR* handle_directory   = 0;
  struct dirent* iterator = 0;
  if ((handle_directory = opendir(directory_.c_str()))) {
    while ((iterator = readdir (handle_directory))) {

      //ds buffer file name
      const std::string file_name = iterator->d_name;

      //ds check if its a left camera image (that is not a hidden linux file)
      if (file_name.find(image_file_name_) != std::string::npos && file_name[0] != '.') {
        image_paths_.push_back(directory_+"/"+file_name);
      }
    }
    closedir(handle_directory);
  } else {
    throw std::runtime_error("invalid image directory");
  }
}
}
