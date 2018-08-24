#include "../../hbst/visualization/closure_viewer.h"

#include "srrg_gl_helpers/opengl_primitives.h"

namespace srrg_bench {

  using namespace srrg_gl_helpers;
  using namespace srrg_core_viewers;

  ClosureViewer::ClosureViewer(const std::string& method_name_): _method_name(method_name_) {
    _image_poses.clear();
    _closure_map.clear();
    _valid_query_image_numbers.clear();
    _valid_train_image_numbers.clear();
    setWindowTitle("closure viewer [OpenGL]");
    setFPSIsDisplayed(true);
  }

  void ClosureViewer::draw() {

    //ds configuration
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glPointSize(1);
    glLineWidth(1);

    //ds set proper viewpoint TODO lock rotation on center
    glPushMatrix();

    //ds draw origin
    glColor3f(0.0, 0.0, 0.0);
    drawAxis(1);

    //ds draw valid closures
    for (std::pair<const ImageWithPose*, const ImageWithPose*> closure: _valid_closures) {
      if (_method_name == "bf") {
        _drawClosure(closure, Eigen::Vector3f(199/255.0,233/255.0,180/255.0));
      } else if (_method_name == "flannlsh") {
        _drawClosure(closure, Eigen::Vector3f(127/255.0,205/255.0,187/255.0));
      } else if (_method_name == "bow") {
        _drawClosure(closure, Eigen::Vector3f(65/255.0,182/255.0,196/255.0));
      } else if (_method_name == "hbst") {
        _drawClosure(closure, Eigen::Vector3f(34/255.0,94/255.0,168/255.0));
      } else {
        _drawClosure(closure, Eigen::Vector3f(0, 1, 0));
      }
    }

    //ds draw first 10000 invalid closures (figure will be qualitatively sufficient)
    uint64_t number_of_drawn_invalid_poses = 0;
    for (std::pair<const ImageWithPose*, const ImageWithPose*> closure: _invalid_closures) {
      if (number_of_drawn_invalid_poses < _maximum_number_of_invalid_closures_to_display) {
        _drawClosure(closure, Eigen::Vector3f(1, 0, 0));
        ++number_of_drawn_invalid_poses;
      } else {
        break;
      }
    }

    //ds draw all poses present
    for (const ImageWithPose* pose: _image_poses) {
      if (pose->image_number%_interspace_images == 0) {
        glPushMatrix();
        glMultMatrixf((pose->pose).cast<float>().data());

        //ds if part of a closure: query
        if (_valid_query_image_numbers.count(pose)) {
          glColor3f(0.0, 1.0, 0.0);

        //ds train
        } else if (_valid_train_image_numbers.count(pose)) {
          glColor3f(0.0, 1.0, 0.0);

        //ds not part of a closure
        } else {
          glColor3f(0.0, 0.0, 0.0);
        }

        //ds draw camera box
        drawPyramidWireframe(1, 1);
        glPopMatrix();
      }
    }
    glPopMatrix();
    glPopAttrib();
  }

  void ClosureViewer::keyPressEvent(QKeyEvent* event_) {
    switch (event_->key()) {
      case Qt::Key_1: {
        _maximum_number_of_invalid_closures_to_display *= 1.5;
        std::cerr << "MapViewer::keyPressEvent|increased maximum number of invalid closures to display: " << _maximum_number_of_invalid_closures_to_display << std::endl;
        break;
      }
      case Qt::Key_2: {
        _maximum_number_of_invalid_closures_to_display *= 0.5;
        std::cerr << "MapViewer::keyPressEvent|decreased maximum number of invalid closures to display: " << _maximum_number_of_invalid_closures_to_display << std::endl;
        break;
      }
      default: {
        SimpleViewer::keyPressEvent(event_);
        break;
      }
    }
  }

  QString ClosureViewer::helpString() const {
      return "See keyboard tab for controls";
  }

  void ClosureViewer::_drawClosure(const std::pair<const ImageWithPose*, const ImageWithPose*>& closure_,
                                   const Eigen::Vector3f& color_rgb_) {

    //ds draw closed frames
    glPushMatrix();
    glMultMatrixf((closure_.first->pose).cast<float>().data());
    glColor3f(color_rgb_.x(), color_rgb_.y(), color_rgb_.z());
    drawPyramidWireframe(3, 3);
    glPopMatrix();
    glPushMatrix();
    glMultMatrixf((closure_.second->pose).cast<float>().data());
    glColor3f(color_rgb_.x(), color_rgb_.y(), color_rgb_.z());
    drawPyramidWireframe(3, 3);
    glPopMatrix();

    //ds draw connections
    glBegin(GL_LINES);
    glColor3f(color_rgb_.x(), color_rgb_.y(), color_rgb_.z());
    const Eigen::Vector3d position_query(closure_.first->pose.translation());
    const Eigen::Vector3d position_reference(closure_.second->pose.translation());
    glVertex3f(position_query.x(), position_query.y(), position_query.z());
    glVertex3f(position_reference.x(), position_reference.y(), position_reference.z());
    glEnd();
  }
}
