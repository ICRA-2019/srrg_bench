
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/

#include "ldahash.h"

/// a.b
float sseg_dot(const float* a, const float* b, int sz )
{
	int ksimdlen = sz/4*4;
	__m128 xmm_a, xmm_b;
	F128   xmm_s;
	float sum;
	int j;
	xmm_s.pack = _mm_set1_ps(0.0);
	for( j=0; j<ksimdlen; j+=4 ) {
  	xmm_a = _mm_loadu_ps((float*)(a+j));
    xmm_b = _mm_loadu_ps((float*)(b+j));
    xmm_s.pack = _mm_add_ps(xmm_s.pack, _mm_mul_ps(xmm_a,xmm_b));
	}
	sum = xmm_s.f[0]+xmm_s.f[1]+xmm_s.f[2]+xmm_s.f[3];
  for(; j<sz; j++ ) sum+=a[j]*b[j];
  	return sum;
}

/// c = Ab
/// A   : matrix
/// ar  : # rows of A
/// ald : # columns of A -> leading dimension as in blas
/// ac  : size of the active part in the row
/// b   : vector with ac size
/// c   : resulting vector with ac size

void sseg_matrix_vector_mul(const float* A, int ar, int ac, int ald, const float* b, float* c)
{
	for( int r=0; r<ar; r++ )
  	c[r] = sseg_dot(A+r*ald, b, ac);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void run_sifthash(const string imname, const int method)
{
	cout << "run_sift " << imname << " " << method << " " << flush;

	IplImage* image;
	if((image = cvLoadImage(imname.c_str(), CV_LOAD_IMAGE_GRAYSCALE)) == 0)
	{
		cout << "could not load " << imname << endl;
		return;
	}

	const int nrRows    = image->height;
	const int nrCols    = image->width;

	VL::PgmBuffer buffer;
	pixel_t* im_pt = new pixel_t [ nrRows*nrCols ];
	pixel_t* start = im_pt;
	
	const uchar *pGray = (uchar*)image->imageData;
	
  for (int i = 0; i < nrRows; i++)
    for (int j = 0; j < nrCols; j++)
      *start++ = float(((uchar*)(pGray + i*image->widthStep))[j*image->nChannels]) / 255.0;

	cvReleaseImage(&image);

	buffer.width  = nrCols;
  buffer.height = nrRows;
  buffer.data   = im_pt;
	
	int    first          = 0 ;
  int    octaves        = 4;
  int    levels         = 3 ;
  float  threshold      = 4.0/100.0 / levels / 2.0f ;
  float  edgeThreshold  = 10.0f;
  float  magnif         = 3.0;
  int    nodescr        = 0 ;
  int    noorient       = 0 ;
  int    stableorder    = 0 ;
  int    savegss        = 0 ;
  int    binary         = 0 ;
  int    haveKeypoints  = 0 ;
  int    unnormalized   = 0 ;
  int    fp             = 0 ;

	int         O      = octaves ;    
  int const   S      = levels ;
  int const   omin   = first ;
  float const sigman = 0.5 ;
  float const sigma0 = 1.6 * powf(2.0f, 1.0f / S) ;
          
  // initialize scalespace
  VL::Sift sift(buffer.data, buffer.width, buffer.height, sigman, sigma0, O, S, omin, -1, S+1) ;
    
  // -------------------------------------------------------------
  //   Run SIFT detector
  // -------------------------------------------------------------    
  
	sift.detectKeypoints(threshold, edgeThreshold);
	
	// -------------------------------------------------------------
  //   Run SIFT orientation detector and descriptor
  // -------------------------------------------------------------    

  /* set descriptor options */
  sift.setNormalizeDescriptor( ! unnormalized ) ;
  sift.setMagnification( magnif ) ;

	const int nrDim  = 128;

  // -------------------------------------------------------------
	//            Run detector, compute orientations and descriptors
	// -------------------------------------------------------------


	image = cvLoadImage(imname.c_str(), CV_LOAD_IMAGE_COLOR);
	CvScalar lineColor    = CV_RGB(0,255,0);

	VL::float_t angles[4];
	vector<SiftDes> siftDVec;
	SiftDes         siftD;
	for( VL::Sift::KeypointsConstIter iter = sift.keypointsBegin(); iter != sift.keypointsEnd(); ++iter) 
	{  
	  // detect orientations
	  int nangles = sift.computeKeypointOrientations(angles, *iter) ;
	  // compute descriptors
		
		cvCircle (image, cvPoint(int(iter->x+0.5),int(iter->y+0.5)), int(iter->sigma*3.0)+1, lineColor);
 	 	for(int a = 0 ; a < nangles ; ++a) 
		{
	    siftD.x  = iter->x;
		  siftD.y  = iter->y;
		  siftD.s  = iter->sigma;
		  siftD.o  = angles[a];

			
			/* compute descriptor */
      sift.computeKeypointDescriptor(siftD.vec, *iter, angles[a]);
			siftDVec.push_back(siftD);
	  }
	}
	cvSaveImage(string(imname+".key.png").c_str(), image);
  cvReleaseImage(&image);

	cout << "run sifthash " << flush;

	BIN_WORD *singleBitWord = new BIN_WORD[64];

  // compute words with particular bit turned on
  // singleBitWord[0] :  000000...000001  <=> 1 = 2^0
  // singleBitWord[1] :  000000...000010  <=> 2 = 2^1
  // singleBitWord[2] :  000000...000100  <=> 4 = 2^2
  // singleBitWord[3] :  000000...001000  <=> 8 = 2^3
	
  singleBitWord[0] = 1;
  for (int i=1; i < 64; i++)
  {
    singleBitWord[i] = singleBitWord[0] << i;
  }
	
  // -------------------------------------------------------------
  //   Run SIFT-Hash
  // -------------------------------------------------------------    

	string na;
	ofstream out;

	int n64;
	int kind;
	if(method == SIFT)     {na = imname + ".sift.key";      n64=128; kind = 1;}
	if(method == DIF128)   {na = imname + ".dif128.key";    n64=2;   kind = 2;}
	if(method == LDA128)   {na = imname + ".lda128.key";    n64=2;   kind = 2;}
	if(method == DIF64)    {na = imname + ".dif64.key";     n64=1;   kind = 2;} 
	if(method == LDA64)    {na = imname + ".lda64.key";     n64=1;   kind = 2;}
	
	const int nrFeat = siftDVec.size();
	out.open(na.c_str());
	out.write((char*)&nrFeat,sizeof(int));
	out.write((char*)&nrFeat,sizeof(int));
	out.write((char*)&n64,sizeof(int));
	out.write((char*)&kind,sizeof(int));
	out.write((char*)&nrRows,sizeof(int));
	out.write((char*)&nrCols,sizeof(int));

	float dum;
	BIN_WORD b;
	float provec[128];
	for(int i=0; i < nrFeat; i++)
	{  
		out.write((char*)&siftDVec[i].x, sizeof(float));
		out.write((char*)&siftDVec[i].y, sizeof(float));
		out.write((char*)&siftDVec[i].s, sizeof(float));
		out.write((char*)&siftDVec[i].o, sizeof(float));
		out.write((char*)&dum,   sizeof(float));

		switch (method) {
			case SIFT : {
				out.write((char*)&siftDVec[i].vec[0],sizeof(float)*128);
				break;
			}			
			case DIF128 : {
				sseg_matrix_vector_mul(Adif128, 128, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k] + tdif128[k] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k+64] + tdif128[k+64] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case LDA128 : {
				sseg_matrix_vector_mul(Alda128, 128, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k] + tlda128[k] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k+64] + tlda128[k+64] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case DIF64 : {
				sseg_matrix_vector_mul(Adif64, 64, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++) {
					if(provec[k] + tdif64[k]  <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case LDA64 : {
				sseg_matrix_vector_mul(Alda64, 64, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++) {
					if(provec[k] + tlda64[k]  <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			delault :{
				cout << "method not known " << endl; exit(0);
			}
	  }
	}
	out.close();
	cout << "nrKeypoints " << nrFeat << endl;
}

void run_sifthash(const string imname, IplImage* mask, const int method)
{
	cout << "run_sift mask " << imname << " " << method << " " << flush;

	IplImage* image;
	if((image = cvLoadImage(imname.c_str(), CV_LOAD_IMAGE_GRAYSCALE)) == 0)
	{
		cout << "could not load " << imname << endl;
		return;
	}
	int n64;
	int kind;
	bool up = false;
	string na;
	if(method == SIFT)     {na = imname + ".sift.key";      n64=128; kind = 1;}
	if(method == DIF128)   {na = imname + ".dif128.key";    n64=2;   kind = 2;}
	if(method == LDA128)   {na = imname + ".lda128.key";    n64=2;   kind = 2;}
	if(method == DIF64)    {na = imname + ".dif64.key";     n64=1;   kind = 2;} 
	if(method == LDA64)    {na = imname + ".lda64.key";     n64=1;   kind = 2;}

	const int nrRows    = image->height;
	const int nrCols    = image->width;

	if(nrRows != mask->height) {cout << "mask rows doe not agree" << endl; exit(0);}
	if(nrCols != mask->width)  {cout << "mask cols doe not agree" << endl; exit(0);} 
	
	VL::PgmBuffer buffer;
	pixel_t* im_pt = new pixel_t [ nrRows*nrCols ];
	pixel_t* start = im_pt;
	
	const uchar *pGray = (uchar*)image->imageData;
	
  for (int i = 0; i < nrRows; i++)
    for (int j = 0; j < nrCols; j++)
      *start++ = float(((uchar*)(pGray + i*image->widthStep))[j*image->nChannels]) / 255.0;

  cvReleaseImage(&image);

	
	buffer.width  = nrCols;
  buffer.height = nrRows;
  buffer.data   = im_pt;
	
	int    first          = 0 ;
  int    octaves        = 4;
  int    levels         = 3 ;
  float  threshold      = 4.0/100.0 / levels / 2.0f ;
  float  edgeThreshold  = 10.0f;
  float  magnif         = 3.0;
  int    nodescr        = 0 ;
  int    noorient       = 0 ;
  int    stableorder    = 0 ;
  int    savegss        = 0 ;
  int    binary         = 0 ;
  int    haveKeypoints  = 0 ;
  int    unnormalized   = 0 ;
  int    fp             = 0 ;

	int         O      = octaves ;    
  int const   S      = levels ;
  int const   omin   = first ;
  float const sigman = 0.5 ;
  float const sigma0 = 1.6 * powf(2.0f, 1.0f / S) ;
          
  // initialize scalespace
  VL::Sift sift(buffer.data, buffer.width, buffer.height, sigman, sigma0, O, S, omin, -1, S+1) ;
    
  // -------------------------------------------------------------
  //   Run SIFT detector
  // -------------------------------------------------------------    
  
	sift.detectKeypoints(threshold, edgeThreshold);
	
	// -------------------------------------------------------------
  //   Run SIFT orientation detector and descriptor
  // -------------------------------------------------------------    

  /* set descriptor options */
  sift.setNormalizeDescriptor( ! unnormalized ) ;
  sift.setMagnification( magnif ) ;

	const int nrDim  = 128;

  // -------------------------------------------------------------
	//            Run detector, compute orientations and descriptors
	// -------------------------------------------------------------

// 	if((image = cvLoadImage(imname.c_str(), CV_LOAD_IMAGE_COLOR)) == 0)
// 	{
// 		cout << "could not load " << imname << endl;
// 		return;
// 	}
// 	CvScalar lineColor    = CV_RGB(255,0,0);

	VL::float_t angles[4];
	vector<SiftDes> siftDVec;
	SiftDes         siftD;
	for( VL::Sift::KeypointsConstIter iter = sift.keypointsBegin(); iter != sift.keypointsEnd(); ++iter) 
	{  
		int ix = int(iter->x+0.5);
		int iy = int(iter->y+0.5);
		if( ((uchar*)(mask->imageData + iy*mask->widthStep))[ix*mask->nChannels] == 0) continue;
	
		if(up == false)
		{
			// detect orientations	
			int nangles = sift.computeKeypointOrientations(angles, *iter) ;
			// compute descriptors
			for(int a = 0 ; a < nangles ; ++a) 
			{
				siftD.x  = iter->x;
				siftD.y  = iter->y;
				siftD.s  = iter->sigma;
				siftD.o  = angles[a];

				/* compute descriptor */
				sift.computeKeypointDescriptor(siftD.vec, *iter, angles[a]);
				siftDVec.push_back(siftD);
			}
		}
		else {
			siftD.x  = iter->x;
			siftD.y  = iter->y;
			siftD.s  = iter->sigma;
			siftD.o  = 0.0;

			/* compute descriptor */
			sift.computeKeypointDescriptor(siftD.vec, *iter, 0.0);
			siftDVec.push_back(siftD);
		}
		//		cvCircle (image, cvPoint(ix,iy), int(iter->sigma*3.0)+1, lineColor);
	}
// 	cvSaveImage(string(imname+".key.png").c_str(), image);
//   cvReleaseImage(&image);
	cout << "run sifthash " << flush;

	BIN_WORD *singleBitWord = new BIN_WORD[64];

  // compute words with particular bit turned on
  // singleBitWord[0] :  000000...000001  <=> 1 = 2^0
  // singleBitWord[1] :  000000...000010  <=> 2 = 2^1
  // singleBitWord[2] :  000000...000100  <=> 4 = 2^2
  // singleBitWord[3] :  000000...001000  <=> 8 = 2^3
	
  singleBitWord[0] = 1;
  for (int i=1; i < 64; i++)
  {
    singleBitWord[i] = singleBitWord[0] << i;
  }
	
  // -------------------------------------------------------------
  //   Run SIFT-Hash
  // -------------------------------------------------------------    

	ofstream out;

	
	const int nrFeat = siftDVec.size();
	out.open(na.c_str());
	out.write((char*)&nrFeat,sizeof(int));
	out.write((char*)&nrFeat,sizeof(int));
	out.write((char*)&n64,sizeof(int));
	out.write((char*)&kind,sizeof(int));
	out.write((char*)&nrRows,sizeof(int));
	out.write((char*)&nrCols,sizeof(int));

	float dum;
	BIN_WORD b;
	float provec[128];
	for(int i=0; i < nrFeat; i++)
	{  
		out.write((char*)&siftDVec[i].x, sizeof(float));
		out.write((char*)&siftDVec[i].y, sizeof(float));
		out.write((char*)&siftDVec[i].s, sizeof(float));
		out.write((char*)&siftDVec[i].o, sizeof(float));
		out.write((char*)&dum,   sizeof(float));

		switch (method) {
			case SIFT : {
				out.write((char*)&siftDVec[i].vec[0],sizeof(float)*128);
				break;
			}			
			case DIF128 : {
				sseg_matrix_vector_mul(Adif128, 128, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k] + tdif128[k] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k+64] + tdif128[k+64] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case LDA128 : {
				sseg_matrix_vector_mul(Alda128, 128, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k] + tlda128[k] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				b = 0;
				for(int k=0; k < 64; k++){
					if(provec[k+64] + tlda128[k+64] <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case DIF64 : {
				sseg_matrix_vector_mul(Adif64, 64, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++) {
					if(provec[k] + tdif64[k]  <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			case LDA64 : {
				sseg_matrix_vector_mul(Alda64, 64, 128, 128, siftDVec[i].vec, provec);
				b = 0;
				for(int k=0; k < 64; k++) {
					if(provec[k] + tlda64[k]  <= 0.0) b |= singleBitWord[k];
				}
				out.write((char*)&b,sizeof(BIN_WORD));
				break;
			}
			delault :{
				cout << "method not known " << endl; exit(0);
			}
	  }
	}
	out.close();
	cout << "nrKeypoints " << nrFeat << endl;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
bool readPoints(const string im, const int method, vector< pair<float,float> > &points)
{
	int nrKeypointsAll = 0;
	int nrKeypointsSim = 0;
	int nrKeypoints    = 0;
	int nrDim          = 0;
	int nrAlg          = 0;
	int nrR            = 0;
	int nrC            = 0;
	float poi[5];
	ifstream in;
	
	
	string task;
	if(method == SIFT)     {task = "sift";}
	if(method == DIF128)   {task = "dif128";}
	if(method == LDA128)   {task = "lda128";}
	if(method == DIF64)    {task = "dif64";} 
	if(method == LDA64)    {task = "lda64";}

	string fileName = im + "." + task + ".key";
	in.open(fileName.c_str());
	if(!in) {
		cout << "could not open " << fileName << endl;
		return false;
	}

	in.read((char*)&nrKeypointsSim, sizeof(int));
	in.read((char*)&nrKeypointsAll, sizeof(int));
	in.read((char*)&nrDim, sizeof(int));
	in.read((char*)&nrAlg, sizeof(int));
	in.read((char*)&nrR, sizeof(int));
	in.read((char*)&nrC, sizeof(int));


	nrKeypoints = nrKeypointsAll;
	
	BIN_WORD *query = new BIN_WORD[nrDim];
	
	points.resize(nrKeypoints);
	for(int i=0; i < nrKeypoints; i++)
	{
		in.read((char*)poi, sizeof(float)*5);
		in.read((char*)query,sizeof(BIN_WORD)*nrDim);
		points[i].first  = poi[0];
		points[i].second = poi[1];
	}
	in.close();
	delete [] query;
	return true;
}
