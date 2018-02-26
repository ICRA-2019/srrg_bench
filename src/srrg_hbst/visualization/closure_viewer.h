#pragma once
#include "srrg_core_viewers/simple_viewer.h"
#include "../utilities/loop_closure_evaluator.h"

namespace srrg_bench {

class ClosureViewer: public srrg_core_viewers::SimpleViewer{
public:

  ClosureViewer(const std::string& method_name_ = "");
  ~ClosureViewer() {_image_poses.clear();
                    _closure_map.clear();
                    _valid_query_image_numbers.clear();
                    _valid_train_image_numbers.clear();
                    _valid_closures.clear();
                    _invalid_closures.clear();}

public:

  //! @brief copy containers
  void update(const std::vector<ImageWithPose*> images_,
              const ClosureMap closure_map_,
              const std::set<const ImageWithPose*> valid_query_image_numbers_,
              const std::set<const ImageWithPose*> valid_train_image_numbers_,
              const ImageNumber& interspace_images_ = 1) {_image_poses = images_;
                                                          _closure_map = closure_map_;
                                                          _valid_query_image_numbers = valid_query_image_numbers_;
                                                          _valid_train_image_numbers = valid_train_image_numbers_;
                                                          _interspace_images = interspace_images_;}

  //! @brief copy containers
  void update(const std::vector<ImageWithPose*> images_,
              const ClosureMap closure_map_,
              std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> valid_closures_,
              std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> invalid_closures_,
              const ImageNumber& interspace_images_ = 1) {_image_poses       = images_;
                                                          _closure_map       = closure_map_;
                                                          _valid_closures    = valid_closures_;
                                                          _invalid_closures  = invalid_closures_;
                                                          _interspace_images = interspace_images_;}

protected:

  virtual void draw();

  virtual void keyPressEvent(QKeyEvent* event_);

  virtual QString helpString() const;

  void _drawClosure(const std::pair<const ImageWithPose*, const ImageWithPose*>& closure_,
                    const Eigen::Vector3f& color_rgb_ = Eigen::Vector3f::Zero());

protected:

  //! @brief currently active image poses
  std::vector<ImageWithPose*> _image_poses;

  //! @brief currently active closure map to draw
  ClosureMap _closure_map;

  //! @brief feasible query image numbers
  std::set<const ImageWithPose*> _valid_query_image_numbers;

  //! @brief feasible train image numbers
  std::set<const ImageWithPose*> _valid_train_image_numbers;

  //! @brief current valid closures
  std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> _valid_closures;

  //! @brief current invalid closures
  std::vector<std::pair<const ImageWithPose*, const ImageWithPose*>> _invalid_closures;

  //! @brief space between subsequent images
  ImageNumber _interspace_images = 1;

  //! @brief method name for which we render the closures (if any)
  std::string _method_name;

  //! @brief maximum number of invalid closures to display
  uint64_t _maximum_number_of_invalid_closures_to_display = 1000;
};
}
