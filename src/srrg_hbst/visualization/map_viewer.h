#pragma once
#include <QtGlobal>
#include <Eigen/Geometry>
#include "srrg_core_viewers/simple_viewer.h"

namespace srrg_bench {

class MapViewer: public srrg_core_viewers::SimpleViewer {

//ds object life
public:

    MapViewer() {setWindowTitle("Trajectory view"); _robots_to_world.clear();};
    ~MapViewer() {};

//ds setters/getters
public:

  void setRotationRobotView(const Eigen::Isometry3d& rotation_robot_view_) {_robot_viewpoint = rotation_robot_view_;}
  void setRobotToWorld(const Eigen::Isometry3d& robot_to_world_) {_robot_to_world = robot_to_world_; _world_to_robot = robot_to_world_.inverse(); _robots_to_world.push_back(robot_to_world_);}

  //ds copying intended
  void setReferencePoses(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> reference_poses_) {_reference_poses = reference_poses_;}
  const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& referencePoses() const {return _reference_poses;}

//ds helpers
public:

  //! @brief Qt standard draw function - LOCKING
  virtual void draw();

  //! @brief Qt help string
  virtual QString helpString() const;

//ds attributes
protected:

  //! @brief display transformation: viewpoint adjustment for ego perspective
  Eigen::Isometry3d _robot_viewpoint = Eigen::Isometry3d::Identity();

  //! @brief current robot position
  Eigen::Isometry3d _world_to_robot = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d _robot_to_world = Eigen::Isometry3d::Identity();

  //! @brief past positions
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> _robots_to_world;

  //! @brief current closed reference poses
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> _reference_poses;
};
}
