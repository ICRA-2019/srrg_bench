#include "viewer_bonsai.h"
#include "srrg_gl_helpers/opengl_primitives.h"

namespace srrg_bench {

  using namespace srrg_gl_helpers;
  using namespace srrg_core_viewers;

  ViewerBonsai::ViewerBonsai(const std::shared_ptr<Tree> tree_,
                             const double& object_scale_,
                             const std::string& window_name_): _tree(tree_),
                                                               _object_scale(object_scale_),
                                                               _window_name(window_name_) {
    _matches.clear();
    setWindowTitle(_window_name.c_str());
    setFPSIsDisplayed(false);

    //ds set viewpoint
    _viewpoint[0][0] = 0;
    _viewpoint[0][1] = 0;
    _viewpoint[0][2] = -1;
    _viewpoint[0][3] = 0;
    _viewpoint[1][0] = 0;
    _viewpoint[1][1] = 1;
    _viewpoint[1][2] = 0;
    _viewpoint[1][3] = 0;
    _viewpoint[2][0] = -1;
    _viewpoint[2][1] = 0;
    _viewpoint[2][2] = 0;
    _viewpoint[2][3] = 0;
    _viewpoint[3][0] = 10;
    _viewpoint[3][1] = 0;
    _viewpoint[3][2] = -50;
    _viewpoint[3][3] = 1;

    //ds set keyboard descriptions
    setKeyDescription(Qt::Key_1, "Toggles tree display");
    setKeyDescription(Qt::Key_2, "Toggles successful query path display");
    setKeyDescription(Qt::Key_Space, "Toggles stepwise/benchmark mode");
  }

  void ViewerBonsai::draw() {

    //ds configuration
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glPointSize(_point_size);
    glLineWidth(_object_scale);

    //ds set proper viewpoint TODO lock rotation on center
    glPushMatrix();
    glMultMatrixd(*_viewpoint);

    //ds draw origin
    glColor3f(0.0, 0.0, 0.0);
    drawAxis(_object_scale);

    //ds from the root
    if (_tree && _tree->root()) {

      //ds root node
      const QVector3D position(0.0, 0.0, 0.0);
      const uint32_t angle_degrees(0);

      //ds draw successful query paths
      if (_option_draw_successful_queries) {
        glBegin(GL_LINES);
        glColor4f(0.0, 0.0, 1.0, 1.0);
        _drawSuccessfulMatches(_matches);
        glEnd();
      }

      //ds recursive function plotting all tree points in their correct locations
      if (_option_draw_tree) {
        glBegin(GL_POINTS);
        _drawNodesRecursive(_tree->root(), position, angle_degrees);
        glEnd();
      }
    }

    glPopMatrix();
    glPopAttrib();
  }

  void ViewerBonsai::keyPressEvent(QKeyEvent* event_) {
    switch(event_->key()) {
      case Qt::Key_1: {
        _option_draw_tree = !_option_draw_tree;
        break;
      }
      case Qt::Key_2: {
        _option_draw_successful_queries = !_option_draw_successful_queries;
        break;
      }
      case Qt::Key_Space: {
        _option_stepwise_playback = !_option_stepwise_playback;

        //ds if we switched back to benchmark - reset the steps
        if (!_option_stepwise_playback) {
          _requested_playback_steps = 0;
        }
        break;
      }
      case Qt::Key_Up: {
        if (_option_stepwise_playback) {
          ++_requested_playback_steps;
        }
        break;
      }
      case Qt::Key_Down: {
        break;
      }
      case Qt::Key_Left: {
        break;
      }
      case Qt::Key_Right: {
        break;
      }
      default: {
        SimpleViewer::keyPressEvent(event_);
        break;
      }
    }
    draw();
    updateGL();
  }

  QString ViewerBonsai::helpString() const {
      return "See keyboard tab for controls";
  }

  void ViewerBonsai::_drawNodesRecursive(const Tree::Node* node_,
                                         const QVector3D& node_position_,
                                         const uint32_t& angle_degrees_) const {

    //ds check if this node has leafs
    if(node_->hasLeafs()) {

      //ds we have leafs - draw this node as a branch (grey) and dispatch function on leafs
      glColor4f(0.5, 0.5, 0.5, 0.25);
      glVertex3d(node_position_.x(), node_position_.y(), node_position_.z());

      //ds leaf positions
      QVector3D position_leaf_ones;
      QVector3D position_leaf_zeroes;

      //ds spacing factor
      const double spacing = _depth_size_per_level*node_->getDepth();

      //ds z is directly proportional to the depth value
      position_leaf_ones.setZ(_spread_size_per_level*spacing);
      position_leaf_zeroes.setZ(position_leaf_ones.z());

      //ds check if we have to fork in x or y
      if (angle_degrees_%2 == 0) {

        //ds same x
        position_leaf_ones.setX(node_position_.x());
        position_leaf_zeroes.setX(node_position_.x());

        //ds fork in y
        position_leaf_ones.setY(node_position_.y()+spacing);
        position_leaf_zeroes.setY(node_position_.y()-spacing);
      } else {

        //ds same y
        position_leaf_ones.setY(node_position_.y());
        position_leaf_zeroes.setY(node_position_.y());

        //ds fork in x
        position_leaf_ones.setX(node_position_.x()+spacing);
        position_leaf_zeroes.setX(node_position_.x()-spacing);
      }

      //ds dispatch function
      _drawNodesRecursive(node_->left, position_leaf_ones, angle_degrees_+1);
      _drawNodesRecursive(node_->right, position_leaf_zeroes, angle_degrees_+1);
    } else {

      //ds no leafs - draw this node as a leaf (GREEN) and terminate recursion
      glColor4f(0.0, 1.0, 0.0, 1.0);
      glVertex3d(node_position_.x(), node_position_.y(), node_position_.z());
    }
  }

  void ViewerBonsai::setMatches(const Tree::MatchVector& matches_) {
    _matches.clear();
    for (Tree::Match match: matches_) {
      _matches.push_back(match);
    }
  }

  void ViewerBonsai::_drawSuccessfulMatches(const Tree::MatchVector& matches_) const {
    for (const Tree::Match& match: matches_) {

      //ds evolved during query propagation
      QVector3D position(0.0, 0.0, 0.0);
      QVector3D position_previous(0.0, 0.0, 0.0);
      uint32_t angle_degrees(0);

      //ds mimic tree matching behavior while simultaneously drawing the path
      const Tree::Node* node_current = _tree->root();
      while (node_current) {

        //ds draw line to previous
        glVertex3d(position.x(), position.y(), position.z());
        glVertex3d(position_previous.x(), position_previous.y(), position_previous.z());
        position_previous = position;

        //ds compute next position spacing
        const double spacing = _depth_size_per_level*node_current->getDepth();
        position.setZ(_spread_size_per_level*spacing);

        //ds if this node has leaves
        if (node_current->hasLeafs()) {

          //ds check the split bit and go deeper
          if (match.matchable_query->descriptor[node_current->indexSplitBit()]) {

            //ds check if we have to fork in x or y
            if (angle_degrees%2 == 0) {
              position.setX(position.x());
              position.setY(position.y()+spacing);
            } else {
              position.setY(position.y());
              position.setX(position.x()+spacing);
            }

            //ds update node and position
            node_current = static_cast<const Tree::Node*>(node_current->right);
            ++angle_degrees;

          } else {

            //ds check if we have to fork in x or y
            if (angle_degrees%2 == 0) {
              position.setX(position.x());
              position.setY(position.y()-spacing);
            } else {
              position.setY(position.y());
              position.setX(position.x()-spacing);
            }

            //ds update node and position
            node_current = static_cast<const Tree::Node*>(node_current->left);
            ++angle_degrees;
          }
        } else {

          //ds cannot go deeper
          break;
        }
      }
    }
  }
}
