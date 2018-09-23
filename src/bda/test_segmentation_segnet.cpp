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

#include <caffe/caffe.hpp>

//ds resolve opencv includes
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <chrono> //Just for time measurement

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

//ds feature handling
#ifndef DESCRIPTOR_SIZE_BITS
  #define DESCRIPTOR_SIZE_BITS 256
#endif
#if CV_MAJOR_VERSION == 2
  cv::Ptr<cv::FeatureDetector> feature_detector         = new cv::FastFeatureDetector(5);
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();
#elif CV_MAJOR_VERSION == 3
  cv::Ptr<cv::FeatureDetector> feature_detector     = cv::FastFeatureDetector::create(5);
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create();
#else
  #error OpenCV version not supported
#endif

class SegNetClassifier {
public:

  SegNetClassifier(const string& model_file, const string& trained_file);
  void predict(const cv::Mat& image);

private:

  void setMean(const string& mean_file);
  void wrapInputLayer(std::vector<cv::Mat>* input_channels);
  void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
  void visualize(Blob<float>* output_layer, const cv::Mat& image_);

private:

  shared_ptr<Net<float> > _net;
  cv::Size _input_geometry;
  int _num_channels;
};

int32_t main(int32_t argc_, char** argv_) {
  if (argc_ != 4) {
    std::cerr << "Usage: " << argv_[0] << " <model.prototxt> <weights.caffemodel> <image>" << std::endl;
    return EXIT_FAILURE;
  }

  //ds initialize google logging
  ::google::InitGoogleLogging(argv_[0]);

  const string file_name_model      = argv_[1];
  const string file_name_weights    = argv_[2];
  const std::string file_name_image = argv_[3];

  std::cerr << "loading classifier with model: " << file_name_model << std::endl;
  std::cerr << "                  and weights: " << file_name_weights << std::endl;
  SegNetClassifier classifier(file_name_model, file_name_weights);
  std::cerr << "loaded classifier" << std::endl;
  std::cerr << "segmenting image: " << file_name_image;

  //ds load image
  const cv::Mat image  = cv::imread(file_name_image);

  //ds label image
  classifier.predict(image);

  //ds done
  return EXIT_SUCCESS;
}


SegNetClassifier::SegNetClassifier(const string& model_file, const string& trained_file) {
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  _net.reset(new Net<float>(model_file, TEST));
  _net->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = _net->input_blobs()[0];
  _num_channels = input_layer->channels();
  CHECK(_num_channels == 3 || _num_channels == 1)
    << "Input layer should have 1 or 3 channels.";
  _input_geometry = cv::Size(input_layer->width(), input_layer->height());
}


void SegNetClassifier::predict(const cv::Mat& image) {
  Blob<float>* input_layer = _net->input_blobs()[0];
  input_layer->Reshape(1, _num_channels,
                       _input_geometry.height, _input_geometry.width);
  /* Forward dimension change to all layers. */
  _net->Reshape();

  std::vector<cv::Mat> input_channels;
  wrapInputLayer(&input_channels);

  preprocess(image, &input_channels);


  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); //Just for time measurement

  _net->ForwardPrefilled();

  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  std::cout << "Processing time = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << " sec" <<std::endl; //Just for time measurement


  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = _net->output_blobs()[0];

  visualize(output_layer, image);
}


void SegNetClassifier::visualize(Blob<float>* output_layer, const cv::Mat& image_) {

  std::cout << "output_blob(n,c,h,w) = " << output_layer->num() << ", " << output_layer->channels() << ", "
        << output_layer->height() << ", " << output_layer->width() << std::endl;

  cv::Mat merged_output_image = cv::Mat(output_layer->height(), output_layer->width(), CV_32F, const_cast<float *>(output_layer->cpu_data()));
  //merged_output_image = merged_output_image/255.0;

  merged_output_image.convertTo(merged_output_image, CV_8U);
  cv::Mat labels(image_.rows, image_.cols, merged_output_image.type());
  cv::resize(merged_output_image, labels, labels.size(), 0, 0);

  cv::Mat labels_display;
  cv::cvtColor(labels.clone(), labels_display, CV_GRAY2BGR);

  //ds compute keypoints in image
  std::vector<cv::KeyPoint> keypoints;
  feature_detector->detect(image_, keypoints);
  cv::Mat image_labeled_keypoints(image_.clone());

  //ds display keypoints with different colors depending on their class
  for (const cv::KeyPoint& keypoint: keypoints) {

    //ds color to be set depending on object class
    cv::Scalar color(0, 0, 0);

    //ds retrieve object class (0, 12)
    const uint32_t object_class = labels.at<uchar>(keypoint.pt);
    switch (object_class) {
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

    //ds draw keypoint
    cv::circle(image_labeled_keypoints, keypoint.pt, 2, color, -1);
  }

  //ds display input image and labeled pixels
  cv::Mat image_display;
  cv::vconcat(image_, image_labeled_keypoints, image_display);
  cv::imshow("test segmentation SegNet", image_display);
  cv::waitKey(0);
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SegNetClassifier::wrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = _net->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void SegNetClassifier::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
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
