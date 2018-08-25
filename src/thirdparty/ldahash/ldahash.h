
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/core/version.hpp>
#include <opencv2/opencv.hpp>

#include "sift.hpp"
#include "sift-conv.tpp"
#include "hashpro.h"

#define BIN_WORD unsigned long long
#define METHOD_SIFT 0
#define DIF128      1
#define LDA128      2
#define DIF64       3
#define LDA64       4

struct SiftDes
{
	float vec[128];
	float x;
	float y;
	float o;
	float s;
};
	
typedef union F128
{
	__m128 pack;
  float f[4];
} F128;

/// a.b
float sseg_dot(const float* a, const float* b, int sz );
void sseg_matrix_vector_mul(const float* A, int ar, int ac, int ald, const float* b, float* c);

void run_sifthash(const cv::Mat& image_, const int method, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& descriptors);
void run_sifthash(const std::string imname, IplImage* mask, const int method);

bool readPoints(const std::string im, const int method,  std::vector< std::pair<float,float> > &points);

//matching functions
void run_hammingmatch(const std::string &im1, const std::string &im2, const int method);
void run_hammingmatch(const std::string &im1, const std::vector<std::string> &im2, const int method);
void run_hammingmatch(const std::vector<BIN_WORD> &binVec, const int nrKey, const std::string &im2, const int method, std::vector< std::pair<unsigned, unsigned> >  &matches);

void saveMatches(const std::vector< std::pair<unsigned, unsigned> > &matches, const char *na);
bool readMatches(std::vector< std::pair<unsigned, unsigned> > &matches, const char *na);
void showMatches(const std::string &im1, const std::string &im2, const int method);
