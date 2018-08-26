
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/

#include "ldahash.h"

using namespace std;

void run_hammingmatch(const vector<BIN_WORD> &binVec, const int nrKeypoints1, const string &im2, const int method, vector< pair<unsigned, unsigned> >  &matches)
{
	int nrKeypoints2;
	int nrKeypointsSin;
	int nrKeypointsAll;
	int nrDim;
	int nrAlg;
	int nrRows;
	int nrCols;
	float poi[5];
	string na;
	ifstream in;
	
	string task;
	if(method == METHOD_SIFT)     {task = "sift";}
	if(method == DIF128)   {task = "dif128";}
	if(method == LDA128)   {task = "lda128";}
	if(method == DIF64)    {task = "dif64";} 
	if(method == LDA64)    {task = "lda64";}
	
	na = im2 + "." + task + ".key";
	in.open(na.c_str());
	if(!in) {
		cout << "could not open " << na << endl;
		return ;
	}

	in.read((char*)&nrKeypointsSin, sizeof(int));
	in.read((char*)&nrKeypointsAll, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrRows, sizeof(int));
	in.read((char*)&nrCols, sizeof(int));

	nrKeypoints2 = nrKeypointsAll;

	vector<BIN_WORD> b(nrDim);
	matches.clear();
	matches.reserve(nrKeypoints1);
	
	for(int i=0; i < nrKeypoints2; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j < nrDim; j++)
		{
			in.read((char*)&b[j],sizeof(BIN_WORD));
		}	
		unsigned minDist1 = 129;
		unsigned minDist2 = 129;
		int bestK         = -1;
		for(int k=0; k < nrKeypoints1; k++)
		{
			unsigned dist = __builtin_popcountll(binVec[k*nrDim] ^ b[0]);
 			for(int j = 1; j < nrDim; j++)
			{
				dist += __builtin_popcountll(binVec[k*nrDim+j] ^ b[j]); 
			}			
			if(dist < minDist1)
			{
				minDist2 = minDist1;
				minDist1 = dist;
				bestK = k;
			}
			else {
				if(dist < minDist2)
				{
					minDist2 = dist;
				}
			}
		}
		if(double(minDist1)/double(minDist2)  < 0.625){
			matches.push_back(pair<unsigned,unsigned>(bestK,i));
		}
  }
	in.close();
}
void run_hammingmatch(const string &im1, const string &im2, const int method)
{
	cout << "run_hammingmatch " << im1 << " " << im2 << " " << flush;
	int nrKeypoints1;
	int nrKeypointsSin;
	int nrKeypointsAll;
	int nrDim;
	int nrAlg;
	int nrRows;
	int nrCols;
	float poi[5];
	string na;
	ifstream in;
	
	string task;
	if(method == METHOD_SIFT)     {task = "sift";}
	if(method == DIF128)   {task = "dif128";}
	if(method == LDA128)   {task = "lda128";}
	if(method == DIF64)    {task = "dif64";} 
	if(method == LDA64)    {task = "lda64";}
	
	na = im1 + "." + task + ".key";
	in.open(na.c_str());
	if(!in) {
		cout << "could not open " << na << endl;
		return;
	}

	in.read((char*)&nrKeypointsSin, sizeof(int));
	in.read((char*)&nrKeypointsAll, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrRows, sizeof(int));
	in.read((char*)&nrCols, sizeof(int));

	nrKeypoints1 = nrKeypointsAll;

	vector<BIN_WORD> binVec(nrKeypoints1*nrDim);	
	for(int i=0; i < nrKeypoints1; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j < nrDim; j++)
		{
			in.read((char*)&binVec[i*nrDim+j],sizeof(BIN_WORD));
		}		
	}
	in.close();

	vector< pair<unsigned, unsigned> >  matches;
	run_hammingmatch(binVec, nrKeypoints1, im2, method, matches);
	
	string im2name = im2.substr(im2.rfind("/",im2.length())+1,im2.length());
	na = im1 + "_" + im2name + "." + task + ".matches";
	saveMatches(matches, na.c_str());		
//	showMatches(im1, im2, method);	
}
void run_hammingmatch(const string &im1, const vector<string> &im2, const int method)
{
	cout << "run_hammingmatch " << im1 << " with " << im2.size() << " " << flush;
	int nrKeypoints1;
	int nrKeypointsSin;
	int nrKeypointsAll;
	int nrDim;
	int nrAlg;
	int nrRows;
	int nrCols;
	float poi[5];
	string na;
	ifstream in;
	
	string task;
	if(method == METHOD_SIFT)     {task = "sift";}
	if(method == DIF128)   {task = "dif128";}
	if(method == LDA128)   {task = "lda128";}
	if(method == DIF64)    {task = "dif64";} 
	if(method == LDA64)    {task = "lda64";}
	
	na = im1 + "." + task + ".key";
	in.open(na.c_str());
	if(!in) {
		cout << "could not open " << na << endl;
		return;
	}

	in.read((char*)&nrKeypointsSin, sizeof(int));
	in.read((char*)&nrKeypointsAll, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrRows, sizeof(int));
	in.read((char*)&nrCols, sizeof(int));

	nrKeypoints1 = nrKeypointsAll;

	vector<BIN_WORD> binVec(nrKeypoints1*nrDim);	
	for(int i=0; i < nrKeypoints1; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j < nrDim; j++)
		{
			in.read((char*)&binVec[i*nrDim+j],sizeof(BIN_WORD));
		}		
	}
	in.close();

	vector< pair<unsigned, unsigned> >  matches;
	for(uint32_t i=0; i < im2.size(); i++)
	{
		run_hammingmatch(binVec, nrKeypoints1, im2[i], method, matches);
	
		string im2name = im2[i].substr(im2[i].rfind("/",im2[i].length())+1,im2[i].length());
		na = im1 + "_" + im2name + "." + task + ".matches";
		saveMatches(matches, na.c_str());
		//showMatches(im1, im2[i], method);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////
void saveMatches(const vector< pair<unsigned, unsigned> > &matches, const char *na)
{
	ofstream out;
	out.open(na);
	out << matches.size() << endl;
	vector< pair<unsigned, unsigned> >::const_iterator iter;
	for(iter = matches.begin(); iter != matches.end(); ++iter)
	{
		out << iter->first << " " << iter->second << endl;
	}
	out.close(); 
}
//////////////////////////////////////////////////////////////////////////////////////////////////
bool readMatches(vector< pair<unsigned, unsigned> > &matches, const char *na)
{
	ifstream in;
	in.open(na);
	if(in){
		int nrMatches;
		in >> nrMatches;
		matches.resize(nrMatches);
		vector< pair<unsigned, unsigned> >::iterator iter;
		for(iter = matches.begin(); iter != matches.end(); ++iter)
		{
			in >> iter->first  >> iter->second;
		}
		in.close();
		return true;
	}
	matches.clear();
	return false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void showMatches(const string &im1, const string &im2, const int method)
{
	string task;
	if(method == METHOD_SIFT)     {task = "sift";}
	if(method == DIF128)   {task = "dif128";}
	if(method == LDA128)   {task = "lda128";}
	if(method == DIF64)    {task = "dif64";} 
	if(method == LDA64)    {task = "lda64";}

	string im2name = im2.substr(im2.rfind("/",im2.length())+1,im2.length());
	string na = im1 + "_" + im2name + "." + task + ".matches";
	cout << "showMatches " << na << endl;

	vector< pair<unsigned, unsigned> > matches;
	if(readMatches(matches, na.c_str()) == false)
	{
		cout << "could not open file " << na << endl; exit(0);
	}

	vector< pair<float,float> > points1;
	vector< pair<float,float> > points2;
	if(readPoints(im1, method, points1) == false){
		cout << "could not open keypoint file " << im1 << endl; return;
	}
	if(readPoints(im2, method, points2) == false){
		cout << "could not open keypoint file " << na << endl; return;
	}
	
	IplImage *imk, *iml, *im;
	CvScalar lineColor = CV_RGB(255,0,0);

	imk = cvLoadImage(im1.c_str(), CV_LOAD_IMAGE_COLOR);
	iml = cvLoadImage(im2.c_str(), CV_LOAD_IMAGE_COLOR);
	im  = cvCreateImage(cvSize(imk->width >= iml->width ? imk->width : iml->width, imk->height+iml->height), imk->depth, 3);

	const uchar *pimk = (uchar*)imk->imageData;
	const uchar *piml = (uchar*)iml->imageData;
	uchar *pim = (uchar*)im->imageData;

	for(int i=0; i < imk->height; i++)
	{
		for(int j=0; j < imk->width; j++)
		{
			pim[i*im->widthStep+j*im->nChannels+0] = pimk[i*imk->widthStep+j*imk->nChannels];
			pim[i*im->widthStep+j*im->nChannels+1] = pimk[i*imk->widthStep+j*imk->nChannels];
			pim[i*im->widthStep+j*im->nChannels+2] = pimk[i*imk->widthStep+j*imk->nChannels];
		}
	}
	for(int i=0; i < iml->height; i++)
	{
		for(int j=0; j < iml->width; j++)
		{
			pim[(i+imk->height)*im->widthStep+j*im->nChannels+0] = piml[i*iml->widthStep+j*iml->nChannels];
			pim[(i+imk->height)*im->widthStep+j*im->nChannels+1] = piml[i*iml->widthStep+j*iml->nChannels];
			pim[(i+imk->height)*im->widthStep+j*im->nChannels+2] = piml[i*iml->widthStep+j*iml->nChannels];
		}
	}
	for(unsigned i=0; i < matches.size(); i++)
	{
		cvLine(im, cvPoint(int(points1[matches[i].first].first + 0.5), int(points1[matches[i].first].second + 0.5)),
							 cvPoint(int(points2[matches[i].second].first + 0.5),int(points2[matches[i].second].second + 0.5)+imk->height), lineColor);
	}
	
	na = im1 + "_" + im2name + "." + task + ".matches.out.png";
	cvSaveImage(na.c_str(),im);
	cvReleaseImage(&im);
	cvReleaseImage(&iml);
	cvReleaseImage(&imk);
}

