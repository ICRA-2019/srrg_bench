#pragma once
#include <QVector3D>
#include <mutex>
#include "srrg_core_viewers/simple_viewer.h"
#include "srrg_hbst_types/binary_tree.hpp"
#include <Eigen/Geometry>

namespace srrg_bench {

class ViewerBonsai: public srrg_core_viewers::SimpleViewer{
public:

  ViewerBonsai(const srrg_hbst::BinaryTree256* tree_ = 0,
               const double& object_scale_ = 1,
               const std::string& window_name_ = "Binary search tree view");

  void configure(std::mutex* mutex_data_transfer_) {_mutex_data_transfer = mutex_data_transfer_;}

  ~ViewerBonsai() {_matches.clear();}

public:

  inline const bool& optionRequested() const {return _option_requested;}
  inline void setOptionRequested(const bool& option_requested_) {_option_requested = option_requested_;}

  void setTree(const srrg_hbst::BinaryTree256* tree_) {_tree = tree_;}
  void setMatches(const srrg_hbst::BinaryTree256::MatchVector& matches_) {
    _matches.clear();
    for (srrg_hbst::BinaryTree256::Match match: matches_) {
      _matches.push_back(match);
    }
  }
  const srrg_hbst::BinaryTree256::MatchVector matches() const {return _matches;}

  inline const bool& optionStepwisePlayback() const {return _option_stepwise_playback;}
  inline const uint32_t& requestedPlaybackSteps() const {return _requested_playback_steps;}
  inline void resetRequestedPlaybackSteps() {_requested_playback_steps = 0;}

protected:

  virtual void draw();

  virtual void keyPressEvent(QKeyEvent* event_);

  virtual QString helpString() const;

  void _drawNodesRecursive(const srrg_hbst::BinaryTree256::Node* node_,
                           const QVector3D& node_position_,
                           const uint32_t& angle_degrees_) const;

  void _drawSuccessfulMatches(const srrg_hbst::BinaryTree256::MatchVector& matches_) const;

protected:

  //! @brief thread communication
  std::mutex* _mutex_data_transfer;
  bool _option_requested = false;

  //! @brief active tree instance
  const srrg_hbst::BinaryTree256* _tree = 0;

  //! @brief active matches from last query
  srrg_hbst::BinaryTree256::MatchVector _matches;

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
