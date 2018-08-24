
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

#include "sift.hpp"
#include "sift-conv.tpp"
#include "hashpro.h"

#define BIN_WORD unsigned long long
#define SIFT     0
#define DIF128   1
#define LDA128   2
#define DIF64    3
#define LDA64    4

using namespace std;
using namespace VL;

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

void run_sifthash(const string imname, const int method);
void run_sifthash(const string imname, IplImage* mask, const int method);

bool readPoints(const string im, const int method,  vector< pair<float,float> > &points);

//matching functions
void run_hammingmatch(const string &im1, const string &im2, const int method);
void run_hammingmatch(const string &im1, const vector<string> &im2, const int method);
void run_hammingmatch(const vector<BIN_WORD> &binVec, const int nrKey, const string &im2, const int method, vector< pair<unsigned, unsigned> >  &matches);

void saveMatches(const vector< pair<unsigned, unsigned> > &matches, const char *na);
bool readMatches(vector< pair<unsigned, unsigned> > &matches, const char *na);
void showMatches(const string &im1, const string &im2, const int method);
