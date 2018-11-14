#pragma once
#include <QVector3D>
#include <mutex>
#include "srrg_core_viewers/simple_viewer.h"
#include "srrg_hbst/types/binary_tree.hpp"
#include <Eigen/Geometry>

//ds HBST setup
#define DESCRIPTOR_SIZE_BITS 256
typedef srrg_hbst::BinaryMatchable<cv::KeyPoint, DESCRIPTOR_SIZE_BITS> Matchable;
typedef srrg_hbst::BinaryNode<Matchable> Node;
typedef srrg_hbst::BinaryTree<Node> Tree;

namespace srrg_bench {

class ViewerBonsai: public srrg_core_viewers::SimpleViewer{
public:

  ViewerBonsai(const std::shared_ptr<Tree> tree_,
               const double& object_scale_ = 1,
               const std::string& window_name_ = "Binary search tree view");

  ~ViewerBonsai() {_matches.clear();}

public:

  void setMatches(const Tree::MatchVector& matches_);
  const Tree::MatchVector matches() const {return _matches;}

  inline const bool& optionStepwisePlayback() const {return _option_stepwise_playback;}
  inline const uint32_t& requestedPlaybackSteps() const {return _requested_playback_steps;}
  inline void decrementRequestedPlaybackSteps() {if (_requested_playback_steps > 0) {--_requested_playback_steps;}}

protected:

  virtual void draw();

  virtual void keyPressEvent(QKeyEvent* event_);

  virtual QString helpString() const;

  void _drawNodesRecursive(const Tree::Node* node_,
                           const QVector3D& node_position_,
                           const uint32_t& angle_degrees_) const;

  void _drawSuccessfulMatches(const Tree::MatchVector& matches_) const;

protected:

  //! @brief active tree instance
  const std::shared_ptr<Tree> _tree = 0;

  //! @brief active matches from last query
  Tree::MatchVector _matches;

  //! @brief display properties
  double _object_scale          = 1;
  double _point_size            = 3;
  double _depth_size_per_level  = 0.1;
  double _spread_size_per_level = 4;
  const std::string _window_name;
  GLdouble _viewpoint[4][4];

  //! @brief options
  bool _option_draw_tree               = true;
  bool _option_draw_successful_queries = true;
  bool _option_stepwise_playback       = true;

  //! @brief stepping buffer
  uint32_t _requested_playback_steps = 0;
};
}
