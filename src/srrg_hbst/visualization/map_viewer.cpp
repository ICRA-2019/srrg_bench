#include "map_viewer.h"
#include "srrg_gl_helpers/opengl_primitives.h"

namespace srrg_bench {

void MapViewer::draw(){

  //ds no specific lighting
  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);

  //ds set ego perspective head
  Eigen::Isometry3d world_to_robot = _robot_viewpoint*_world_to_robot;
  glPushMatrix();
  glMultMatrixf(world_to_robot.cast<float>().data());
  glLineWidth(1);

  //ds draw closed frames in green
  for (const Eigen::Isometry3d& robot_to_world: _reference_poses) {
    glPushMatrix();
    glMultMatrixf(robot_to_world.cast<float>().data());
    glColor3f(0, 1, 0);

    //ds draw camera box
    srrg_gl_helpers::drawPyramidWireframe(0.25, 0.25);
    glPopMatrix();
  }

  //ds draw lines from current frame to closed ones
  glBegin(GL_LINES);
  for (const Eigen::Isometry3d& robot_to_world: _reference_poses) {
    glColor3f(0, 1, 0);
    glVertex3f(_robot_to_world.translation().x(), _robot_to_world.translation().y(), _robot_to_world.translation().z());
    glVertex3f(robot_to_world.translation().x(), robot_to_world.translation().y(), robot_to_world.translation().z());
  }
  glEnd();

  //ds draw trajectory in blue
  for (const Eigen::Isometry3d& robot_to_world: _robots_to_world) {
    glPushMatrix();
    glMultMatrixf(robot_to_world.cast<float>().data());
    glColor3f(0, 0, 1);

    //ds draw camera box
    srrg_gl_helpers::drawPyramidWireframe(0.25, 0.25);
    glPopMatrix();
  }
  glPopMatrix();
}

QString MapViewer::helpString( ) const {
    return "See 'Keyboard' tab for controls";
}
}
