/*
 *
 *   Created on: Jan 21, 2017
 *       Author: Timo Sämann
 *  Modified by: Dominik Schlegel
 *
 *  The basis for the creation of this script was the classification.cpp example (caffe/examples/cpp_classification/classification.cpp)
 *
 *  This script visualize the semantic segmentation for your input image.
 *
 *  To compile this script you can use a IDE like Eclipse. To include Caffe and OpenCV in Eclipse please refer to
 *  http://tzutalin.blogspot.de/2015/05/caffe-on-ubuntu-eclipse-cc.html
 *  and http://rodrigoberriel.com/2014/10/using-opencv-3-0-0-with-eclipse/ , respectively
 *
 *
 */

#pragma once
#include <caffe/caffe.hpp>

//ds resolve opencv includes
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

#if CV_MAJOR_VERSION == 2
  //ds no specifics
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace srrg_bench {

class SegNetClassifier {
public:

  SegNetClassifier(const std::string& model_file, const std::string& trained_file) {
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    /* Load the network. */
    _net.reset(new caffe::Net<float>(model_file, caffe::TEST));
    _net->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

    caffe::Blob<float>* input_layer = _net->input_blobs()[0];
    _num_channels = input_layer->channels();
    CHECK(_num_channels == 3 || _num_channels == 1)
      << "Input layer should have 1 or 3 channels.";
    _input_geometry = cv::Size(input_layer->width(), input_layer->height());
  }

  cv::Mat getImageWithLabels(const cv::Mat& image) {
    caffe::Blob<float>* input_layer = _net->input_blobs()[0];
    input_layer->Reshape(1, _num_channels,
                         _input_geometry.height, _input_geometry.width);
    /* Forward dimension change to all layers. */
    _net->Reshape();

    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels);

    preprocess(image, &input_channels);

    _net->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    caffe::Blob<float>* output_layer = _net->output_blobs()[0];

    cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
    //merged_output_image = merged_output_image/255.0;

    merged_output_image.convertTo(merged_output_image, CV_8U);
    cv::Mat image_with_labels(image.rows, image.cols, merged_output_image.type());
    cv::resize(merged_output_image, image_with_labels, image_with_labels.size(), 0, 0);
    return image_with_labels;
  }

  cv::Scalar getColorForLabel(const uint32_t& label_) const {
    cv::Scalar color(0, 0, 0);
    switch (label_) {
      case 0: { //Sky
        color = cv::Scalar(128, 128, 128);
        break;
      }
      case 1: { //Building
        color = cv::Scalar(0, 0, 128);
        break;
      }
      case 2: { //Pole
        color = cv::Scalar(128, 192, 192);
        break;
      }
      case 3: { //Road (1)
        color = cv::Scalar(128, 64, 128);
        break;
      }
      case 4: { //Road (2)
        color = cv::Scalar(128, 64, 128);
        break;
      }
      case 5: { //Pavement
        color = cv::Scalar(222, 40, 60);
        break;
      }
      case 6: { //Tree
        color = cv::Scalar(0, 128, 128);
        break;
      }
      case 7: { //Sign
        color = cv::Scalar(128, 128, 192);
        break;
      }
      case 8: { //Fence
        color = cv::Scalar(128, 64, 64);
        break;
      }
      case 9: { //Vehicle
        color = cv::Scalar(128, 0, 64);
        break;
      }
      case 10: { //Pedestrian
        color = cv::Scalar(0, 64, 64);
        break;
      }
      case 11: { //Bike
        color = cv::Scalar(192, 128, 0);
        break;
      }
      default: {
        color = cv::Scalar(0, 0, 0);
      }
    }
    return color;
  }

private:

  /* Wrap the input layer of the network in separate cv::Mat objects
   * (one per channel). This way we save one memcpy operation and we
   * don't need to rely on cudaMemcpy2D. The last preprocessing
   * operation will write the separate channels directly to the input
   * layer. */
  void wrapInputLayer(std::vector<cv::Mat>* input_channels) {
    caffe::Blob<float>* input_layer = _net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
  }

  void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && _num_channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && _num_channels == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && _num_channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && _num_channels == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    if (sample.size() != _input_geometry)
      cv::resize(sample, sample_resized, _input_geometry);
    else
      sample_resized = sample;

    cv::Mat sample_float;
    if (_num_channels == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == _net->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
  }

private:

  std::shared_ptr<caffe::Net<float> > _net;
  cv::Size _input_geometry;
  int _num_channels;
};
}
