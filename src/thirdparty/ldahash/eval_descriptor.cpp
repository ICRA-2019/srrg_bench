
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <map>
#include <sys/stat.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

#define BIN_WORD unsigned long long

struct point 
{
	float x;
	float y;
	float s;
	float o;
	int ind;
};
struct pointMat 
{
	int ind1;
	int ind2;
};

using namespace std;

bool readPointsFromKey(const char *fileName, vector<point> *points, int *nrRows = NULL, int *nrCols = NULL);

int readKeyPoints(const char *fileName, vector<BIN_WORD> *des);
int readKeyPoints(const char *fileName, vector<float> *des);

void eval(string fileName1, string fileName2, string tsak);
void ground(string fileName1, string fileName2, string tsak);
void makeFlow();

float* readFlow(const char *fileName, int *nrRows, int *nrCols);

map<int,int> match01;
map<int,int> match10;

vector<int> Gmatch01;
vector<int> Gmatch10;
	
int main(int argc, char **argv) 
{

  vector<string> fileNameList;
	string task;
	
  int i;
  // loop over arguments, start with i=1 because program-name is in argv[0]
  for( i = 1 ; (i < argc) && ((argv[i])[0] == '-') ; i++)
  {
  	switch ((argv[i])[1])
		{
			case 'i':
	  		i++;
	  		while((i < argc) && ((argv[i])[0] != '-'))
	    	{
	      	string a = argv[i]; fileNameList.push_back(a);i++;
	    	}
	  		i--;
	  		break;
			case 't':
	  		i++; task = argv[i]; break;
			default:
	  		cout << endl << "Unrecognized option " << argv[i] << endl;
	  		exit(0);
		}
  }

	cout << "eval " << fileNameList.size() << endl;

	string fileName1;
	string fileName2;
	
	for(int i=1; i < fileNameList.size(); i++)
	{
		fileName1 = fileNameList[i];
		fileName2 = fileNameList[0];
			
		match01.clear();
		match10.clear();
			
		ground(fileName1, fileName2, task);
		
		eval(fileName1, fileName2, task);
	}	
}

void eval(string fileName1, string fileName2, string task)
{
	cout << "eval images " << fileName1 << " " << fileName2 << " " << task << endl;
	
	vector<double> truePos;
	vector<double> falsePos;
	
	string na1 = fileName1 + "." + task + ".key";
	string na2 = fileName2 + "." + task + ".key";
	
	
	map<float,int> distMap;
	vector<BIN_WORD> binVec1;
	vector<BIN_WORD> binVec2;
	vector<float>    floatVec1;
	vector<float>    floatVec2;

	int nrDim           = 0;
	int nrHist          = 1000;
	int nrGroundMatches = 0;
		
	if(task == string("sift")) //load floating points vectors
	{
		nrDim = readKeyPoints(na1.c_str(), &floatVec1);
		if(readKeyPoints(na2.c_str(), &floatVec2) != nrDim){
			cout << "dimension do not agree " << endl; exit(0);
		}
		
		//compute maxDist over ground truth matches
		float maxDist = 0;
		for(int i1=0; i1 < Gmatch01.size(); i1++)
		{
			int i2 = Gmatch01[i1];
			if(i2 == -1) continue;
			nrGroundMatches++;
			float dist = 0.0;
			for(int j=0; j < nrDim; j++)
			{
				float d = floatVec1[i1*nrDim+j] - floatVec2[i2*nrDim+j];
				dist += d*d;
			}
			if(dist > maxDist) maxDist = dist;
		}
		float scale = float(nrHist-1)/maxDist;
		cout << "maxDist " << maxDist << " nrGroundmatches " << nrGroundMatches << " scale " << scale << endl;
		
		//compute ROC staristics
		truePos.resize(nrHist);
		falsePos.resize(nrHist);
		for(int i=0; i < nrHist; i++) {
			truePos[i] = 0.0;
			falsePos[i] = 0.0;
		}
		
		for(int i1=0; i1 < Gmatch01.size(); i1++)
		{
			if(Gmatch01[i1] == -1) continue;
			for(int i2=0; i2 < Gmatch10.size(); i2++)
			{
				if(Gmatch10[i2] == -1) continue;
				
				float fdist = 0.0;
				for(int j=0; j < nrDim; j++)
				{
					float d = floatVec1[i1*nrDim+j] - floatVec2[i2*nrDim+j];
					fdist += d*d;
				}
				int dist = fdist*scale;
				if(Gmatch01[i1] == i2){
					if(Gmatch10[i2] != i1) {
						cout << "error match " << endl; exit(0);
					}
					for(int k=dist; k < nrHist; k++)
					{
						truePos[k]++;
					}
				}
				else { 
					for(int k=dist; k < nrHist; k++)
					{
						falsePos[k]++;
					}
				}
			}
		}
	}
	else {
		int nrDim = readKeyPoints(na1.c_str(), &binVec1);
		if(readKeyPoints(na2.c_str(), &binVec2) != nrDim){
			cout << "dimension do not agree " << endl; exit(0);
		}
		//compute maxDist over ground truth matches
		int maxDist = 0;
		for(int i1=0; i1 < Gmatch01.size(); i1++)
		{
			int i2 = Gmatch01[i1];
			if(i2 == -1) continue;
			nrGroundMatches++;
			int dist = 0;
			for(int j=0; j < nrDim; j++)
			{
				dist += __builtin_popcountll(binVec1[i1*nrDim+j] ^ binVec2[i2*nrDim+j]);
			}
			if(dist > maxDist) maxDist = dist;
		}
		
		nrHist = maxDist + 1;
		cout << "maxDist " << maxDist << " nrGroundmatches " << nrGroundMatches << endl;
		
		//compute ROC staristics
		truePos.resize(nrHist);
		falsePos.resize(nrHist);
		for(int i=0; i < nrHist; i++) {
			truePos[i] = 0.0;
			falsePos[i] = 0.0;
		}
		
		for(int i1=0; i1 < Gmatch01.size(); i1++)
		{
			if(Gmatch01[i1] == -1) continue;
			for(int i2=0; i2 < Gmatch10.size(); i2++)
			{
				if(Gmatch10[i2] == -1) continue;
				
				int dist = 0;
				for(int j=0; j < nrDim; j++)
				{
					dist += __builtin_popcountll(binVec1[i1*nrDim+j] ^ binVec2[i2*nrDim+j]);
				}
				if(Gmatch01[i1] == i2){
					if(Gmatch10[i2] != i1) {
						cout << "error match " << endl; exit(0);
					}
					for(int k=dist; k < nrHist; k++)
					{
						truePos[k]++;
					}
				}
				else { 
					for(int k=dist; k < nrHist; k++)
					{
						falsePos[k]++;
					}
				}
			}
		}
	}
	ofstream out;
	string na = fileName1 + "." + task + ".roc";
	out.open(na.c_str());
	for(int i=0; i < nrHist; i++)
	{
		out << falsePos[i]/double((nrGroundMatches)*(nrGroundMatches-1)) << " " << truePos[i]/double(nrGroundMatches) << endl;
	}
	out.close();
		
	na = fileName1 + "." + task + ".rp";
	out.open(na.c_str());
	for(int i=0; i < nrHist; i++)
	{
		out << truePos[i] / (truePos[i] + falsePos[i] + 0.00000001) << " " << truePos[i]/double(nrGroundMatches) << " " << endl;
	}
	out.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int readKeyPoints(const char *fileName, vector<BIN_WORD> *des)
{
	int nrKeypoints;
	int nrDim;
	int nrAlg;
	int nrRows;
	int nrCols;
	float poi[5];
	
	ifstream in;
	in.open(fileName);
	if(!in) {
		cout << "could not open " << fileName << endl;
		return false;
	}

	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrRows, sizeof(int));
	in.read((char*)&nrCols, sizeof(int));

	const int n64  = nrDim;
	const float nr = nrRows;
	const float nc = nrCols;

	cout << fileName << " " << nrKeypoints << " " << nrDim << " " << n64 << " " << nrAlg << " " << nrRows << " " << nrCols << " ";

	BIN_WORD b;
	des->resize(nrKeypoints*n64);
	int k=0;
	for(int i=0; i < nrKeypoints; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j< n64; j++)
		{
			in.read((char*)&b,sizeof(BIN_WORD));
			(*des)[k] = b; k++;
		}
	}
	in.close();
	cout << "nrKey : " << nrKeypoints << endl;
	return n64;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int readKeyPoints(const char *fileName, vector<float> *des)
{
	int nrKeypoints;
	int nrDim;
	int nrAlg;
	int nrRows;
	int nrCols;
	float poi[5];
	
	ifstream in;
	in.open(fileName);
	if(!in) {
		cout << "could not open " << fileName << endl;
		return false;
	}

	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrRows, sizeof(int));
	in.read((char*)&nrCols, sizeof(int));

	const int n64  = nrDim;
	const float nr = nrRows;
	const float nc = nrCols;

	cout << fileName << " " << nrKeypoints << " " << nrDim << " " << n64 << " " << nrAlg << " " << nrRows << " " << nrCols << " " << endl;

	des->resize(nrKeypoints*n64);
	float b;
	int k=0;
	for(int i=0; i < nrKeypoints; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j< n64; j++)
		{
			in.read((char*)&b,sizeof(float));
			(*des)[k] = b; k++;
		}
	}
	in.close();
	cout << "nrKey : " << nrKeypoints << endl;
	return n64;
}
//////////////////////////////////////////////////////////////////////////////////////
bool readPointsFromKey(const char *fileName, vector<point> *points, int *nrRows, int *nrCols)
{
	int nrKeypoints;
	int nrDim;
	int nrAlg;
	int nrR;
	int nrC;
	float poi[5];

	ifstream in;
	in.open(fileName);
	if(!in) {
		cout << "could not open " << fileName << endl;
		return false;
	}

	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrR, sizeof(int));
	in.read((char*)&nrC, sizeof(int));

	if(nrAlg == 2) // binary
	{
		BIN_WORD query;
		points->resize(nrKeypoints);
		for(int i=0; i < nrKeypoints; i++)
		{
			in.read((char*)poi, sizeof(float)*5);
			for(int j=0; j < nrDim; j++) in.read((char*)&query,sizeof(BIN_WORD));
			(*points)[i].x = poi[0];
			(*points)[i].y = poi[1];
			(*points)[i].s = poi[2];
			(*points)[i].o = poi[3];
		}
	}
	else {
		float *query = new float[nrDim];
		points->resize(nrKeypoints);
		for(int i=0; i < nrKeypoints; i++)
		{
			in.read((char*)poi, sizeof(float)*5);
			in.read((char*)query,sizeof(float)*nrDim);
			(*points)[i].x = poi[0];
			(*points)[i].y = poi[1];
			(*points)[i].s = poi[2];
			(*points)[i].o = poi[3];
		}
		delete [] query;
	}
	in.close();
	if(nrRows != NULL) *nrRows = nrR;
	if(nrCols != NULL) *nrCols = nrC;
	return true;
}
//////////////////////////////////////////////////////////////////////////////////////
bool readPointsFromKeyBin(const char *fileName, vector<point> *points, int *nrRows, int *nrCols)
{
	int nrKeypoints;
	int nrDim;
	int nrAlg;
	int nrR;
	int nrC;
	float poi[5];
	ifstream in;
	in.open(fileName);
	if(!in) {
		cout << "could not open " << fileName << endl;
		return false;
	}

	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrKeypoints, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrR, sizeof(int));
	in.read((char*)&nrC, sizeof(int));

	BIN_WORD query;
	points->resize(nrKeypoints);
	for(int i=0; i < nrKeypoints; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		for(int j=0; j < nrDim; j++) in.read((char*)&query,sizeof(BIN_WORD));
		(*points)[i].x = poi[0];
		(*points)[i].y = poi[1];
		(*points)[i].s = poi[2];
		(*points)[i].o = poi[3];
	}
	in.close();
	if(nrRows != NULL) *nrRows = nrR;
	if(nrCols != NULL) *nrCols = nrC;
	return true;
}

void ground(string fileName1, string fileName2, string task)
{
	cout << "ground truth for " << fileName1 << " " << fileName2 << " " << task << endl;
	string imageName1  = fileName1.substr(fileName1.rfind("/",fileName1.length())+1,fileName1.length());
	string imageName2  = fileName2.substr(fileName2.rfind("/",fileName2.length())+1,fileName2.length());
	string naStr;
	IplImage *vis01, *vis10;
	IplImage *im01, *im10;
	vector<point> points1;
	vector<point> points2;	
	int nrR, nrC;

	naStr = fileName1 + "." + imageName2 + ".vis.png";
	cout << naStr << endl;
	vis01 = cvLoadImage(naStr.c_str(), CV_LOAD_IMAGE_COLOR);
	im01  = cvLoadImage(fileName1.c_str(), CV_LOAD_IMAGE_COLOR);
	
	naStr = fileName2 + "." + imageName1 + ".vis.png";
	cout << naStr << endl;
	vis10 = cvLoadImage(naStr.c_str(), CV_LOAD_IMAGE_COLOR);
	im10  = cvLoadImage(fileName2.c_str(), CV_LOAD_IMAGE_COLOR);

	naStr = fileName1 + "." + task + ".key";
	cout << naStr << endl;
	readPointsFromKey(naStr.c_str(), &points1, &nrR, &nrC);
	naStr = fileName2 + "." + task + ".key";
	cout << naStr << endl;
	readPointsFromKey(naStr.c_str(), &points2, &nrR, &nrC);
	
	cout << "task " << task << " " << points1.size() << " " << points2.size() << endl;
	
	map<int, point> map01;
	map<int, point> map10;
	point p;
	CvScalar colorRed   = CV_RGB(255,0,0);
	CvScalar colorGreen = CV_RGB(0,255,0);
	
	for(int i=0; i < points1.size(); i++)
	{
		float x = points1[i].x;
		float y = points1[i].y;
		float s = points1[i].s;
		float o = points1[i].o;
		int ix = int(x+0.5);
		int iy = int(y+0.5);
		if(((uchar*)vis01->imageData + iy*vis01->widthStep)[ix*vis01->nChannels+2] != 255 &&
		   ((uchar*)vis01->imageData + (iy+1)*vis01->widthStep)[ix*vis01->nChannels+2] != 255 &&
		   ((uchar*)vis01->imageData + (iy+1)*vis01->widthStep)[(ix+1)*vis01->nChannels+2] != 255 &&
		   ((uchar*)vis01->imageData + (iy)*vis01->widthStep)[(ix+1)*vis01->nChannels+2] != 255)
		{
			p.x = x; p.y = y; p.ind = i; p.s = s; p.o = o;
			map01[i] = p;
			cvCircle(im01, cvPoint(ix,iy), int(rint(s))+1, colorGreen);
		}
		else {
			cvCircle(im01, cvPoint(ix,iy), int(rint(s))+1, colorRed);
		}
	}
	
	for(int i=0; i < points2.size(); i++)
	{
		float x = points2[i].x;
		float y = points2[i].y;
		float s = points2[i].s;
		float o = points2[i].o;
		int ix = int(x+0.5);
		int iy = int(y+0.5);
		if(((uchar*)vis10->imageData + iy*vis10->widthStep)[ix*vis10->nChannels+2] != 255 &&
		   ((uchar*)vis10->imageData + (iy+1)*vis10->widthStep)[ix*vis10->nChannels+2] != 255 &&
		   ((uchar*)vis10->imageData + (iy+1)*vis10->widthStep)[(ix+1)*vis10->nChannels+2] != 255 &&
		   ((uchar*)vis10->imageData + (iy)*vis10->widthStep)[(ix+1)*vis10->nChannels+2] != 255)
		{
			p.x = x; p.y = y; p.ind = i; p.s = s; p.o = o;
			map10[i] = p;
			cvCircle(im10, cvPoint(ix,iy), int(rint(s))+1, colorGreen);
		}
		else {
			cvCircle(im10, cvPoint(ix,iy), int(rint(s))+1, colorRed);
		}
	}
	cout << "vis" << endl;
	
	naStr = fileName2 + "." + imageName1 + ".points.png";
	cout << naStr << endl;
	cvSaveImage(naStr.c_str(), im01);
	cvReleaseImage(&im01);	
	naStr = fileName1 + "." + imageName2 + ".points.png";
	cout << naStr << endl;
	cvSaveImage(naStr.c_str(), im10);
	cvReleaseImage(&im10);
	
	naStr = fileName1 + "." + imageName2 + ".flow.dat";
	int nrRows,nrCols;
	float *flow = readFlow(naStr.c_str(), &nrRows, &nrCols);
	cout << "flow " << nrRows << " " << nrCols << " " << flow << endl;
	map<int, point>::iterator map01iter;
	map<int, point>::iterator map10iter;

	map<int,vector< int> > matchVec01;
	map<int,vector< int> > matchVec10;

	for(map01iter = map01.begin(); map01iter != map01.end(); ++map01iter)
	{
		float x = map01iter->second.x;
		float y = map01iter->second.y;
		
		int ix   = int(x);
		int iy   = int(y);
		if(!(ix >= 0 && ix <= 3070 && iy >= 0 && iy <= 2046)) continue;
			
		float dc = x-float(ix);
		float dr = y-float(iy);
		
		float ixt = (1.0-dc)*(1.0-dr)*flow[iy*nrCols*2+ix*2+0] + 
								(dc    )*(1.0-dr)*flow[iy*nrCols*2+(ix+1)*2+0]+ 
								(dc    )*(    dr)*flow[(iy+1)*nrCols*2+(ix+1)*2+0]+ 
								(1.0-dc)*(dr    )*flow[(iy+1)*nrCols*2+ix*2+0];
		float iyt = (1.0-dc)*(1.0-dr)*flow[iy*nrCols*2+ix*2+1] + 
								(dc    )*(1.0-dr)*flow[iy*nrCols*2+(ix+1)*2+1]+ 
								(dc    )*(    dr)*flow[(iy+1)*nrCols*2+(ix+1)*2+1]+ 
								(1.0-dc)*(dr    )*flow[(iy+1)*nrCols*2+ix*2+1];
		
		float minDist = 5.0;
		int i1, i2;
		for(map10iter = map10.begin(); map10iter != map10.end(); ++map10iter)
		{	
			float dx = ixt - map10iter->second.x;
			float dy = iyt - map10iter->second.y;
			float d = sqrt(dx*dx+dy*dy);
			if(d <= minDist)
			{
				i1 = map01iter->second.ind;
				i2 = map10iter->second.ind;
				minDist = d;
			}
		}
		if(minDist <= 2.0)
		{
//			cout << "match " << map01[i1].x << " " << map01[i1].y << " -> " << ixt << " " << iyt << " = ";
//			cout << map10[i2].x << " " << map10[i2].y << " " << minDist << endl;
			matchVec01[i1].push_back(i2);
			matchVec10[i2].push_back(i1);
		}
	}
	map<int,vector<int> >::iterator matchVecIter;
	for(matchVecIter = matchVec01.begin(); matchVecIter != matchVec01.end(); ++matchVecIter)
	{
		if(matchVecIter->second.size() == 1)
		{
			if(matchVec10[matchVecIter->second[0]].size() == 1)
			{
				if(matchVec10[matchVecIter->second[0]][0] == matchVecIter->first)
				{
					match01[matchVecIter->first] = matchVecIter->second[0];
					match10[matchVecIter->second[0]] = matchVecIter->first;
				}
			}
		}
	}
	
// 	vector< pair<int,int> > matches;
// 	vector< pair<int,int> > allmatches;
// 	map<int,int>::iterator matchIter;
// 	for(matchIter = match01.begin(); matchIter != match01.end(); ++matchIter)
// 	{
// 		allmatches.push_back(pair<int,int>(matchIter->first,matchIter->second));
// 		matches.push_back(pair<int,int>(matchIter->first,matchIter->second));
// 	}
// 	
// 	cout << "all matches " << allmatches.size() << endl;
// 	
// 	int nrReqMatches = 5000;
// 	while(matches.size() < nrReqMatches)
// 	{
// 		double rr = rand() / double(RAND_MAX);
// 		int r = int(double(allmatches.size()) * rr);
// 		if(r >= 0 && r < allmatches.size())
// 		{
// 			if(allmatches[r].first == -1) continue;
// 			matches.push_back(allmatches[r]);
// 			allmatches[r].first = -1;
// 		}
// 	}
// 	
// 	cout << "matches " << matches.size() << " " << nrReqMatches << endl;

	vector< pair<int,int> > matches;
	map<int,int>::iterator matchIter;
	for(matchIter = match01.begin(); matchIter != match01.end(); ++matchIter)
	{
		matches.push_back(pair<int,int>(matchIter->first,matchIter->second));
	}
	
	
	Gmatch01.resize(points1.size());
	Gmatch10.resize(points2.size());
	
	for(int i=0; i < Gmatch01.size(); i++) Gmatch01[i] = -1;
	for(int i=0; i < Gmatch10.size(); i++) Gmatch10[i] = -1;
	
	for(int i=0; i < matches.size(); i++)
	{
		//cout << i << " " << matches[i].first << " " << matches[i].second << endl;
		Gmatch01[matches[i].first]  = matches[i].second;
		Gmatch10[matches[i].second] = matches[i].first;
	}
// 	for(matchIter = match01.begin(); matchIter != match01.end(); ++matchIter)
// 	{
// 		Gmatch01[matchIter->first]  = matchIter->second;
// 		Gmatch10[matchIter->second] = matchIter->first;
// 	}
}


float* readFlow(const char *fileName, int *nrRows, int *nrCols)
{
	cout << fileName << endl;
	ifstream in;
	in.open(fileName);
	
	unsigned int type;      // 0:byte ,1:float 2:double
  unsigned int nr;
  unsigned int nc;
  unsigned int nb;
  in.read((char*)&type, sizeof(unsigned int));
  in.read((char*)&nr  , sizeof(unsigned int));
  in.read((char*)&nc  , sizeof(unsigned int));
  in.read((char*)&nb  , sizeof(unsigned int));
  if(type != 1)
  {
    cout << "ifstream& operator>> (ifstream& ifs, CR_FloatImage& m)";
    exit(0);
  }

	float *flow = new float[nr*nc*2];
  // read data
  in.read((char*)flow, (size_t)sizeof(float)*nr*nc*2);
	in.close();
	*nrRows = nr;
	*nrCols = nc;
	return flow;
}

