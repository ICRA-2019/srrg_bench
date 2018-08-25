
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/

#include "ldahash.h"

using namespace std;

int main(int argc, char **argv) 
{
  vector<string> fileNameList;
  string fileNameFile;
  string task = "dif128";
	bool showhelp = false;
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
			case 'c':
			{
	  		i++;
				fileNameFile = argv[i]; 
	    	break;
			}
			case 't':
			{
	  		i++;
				task = argv[i]; 
	    	break;
			}
			case 'h':
			{
	  		i++;
				showhelp = true;; 
	    	break;
			}
			default:
	  		cout << endl << "Unrecognized option " << argv[i] << endl;
	  		exit(0);
		}
  }

 
	if(fileNameFile.length() >= 1)
	{
		ifstream in;
		in.open(fileNameFile.c_str());
		string a;
		while(!in.eof())
		{
			in >> a; 
			if(in.eof()) continue;
			fileNameList.push_back(a);
		}
		in.close();
	}

	if(fileNameList.size() <= 0) showhelp = true;
	
	if(showhelp) 
  {    
      cout << "run_ldahash 1.0 Christoph Strecha 2010 christoph.strecha@epfl.ch" << endl;
      cout << "Usage: run_ldahash -i fileName [options]" << endl;
      cout << "  options:" << endl;
      cout << "    -t              dif128 , dif64 or sift will compute 128-bit, 64-bit or sift descriptors" << endl;
      cout << "    -h              this help" << endl << endl;
      cout << "Example:" << endl;
      cout << "    run_ldahash -i image.jpg -t dif64" << endl;
      cout << "    will compute 64-bit descriptors for image.jpg" << endl;
      exit(0);	
  }

	for(size_t i=0; i < fileNameList.size(); i++)
	{
	  //ds buffers
	  std::vector<cv::KeyPoint> keypoints(0);
	  cv::Mat descriptors;

	  //ds load image
    cv::Mat image = cv::imread(fileNameList[i], CV_LOAD_IMAGE_GRAYSCALE);

		if(task == string("dif128")) run_sifthash(image, DIF128, keypoints, descriptors);
		if(task == string("lda128")) run_sifthash(image, LDA128, keypoints, descriptors);
		if(task == string("dif64"))  run_sifthash(image, DIF64, keypoints, descriptors);
		if(task == string("lda64"))  run_sifthash(image, LDA64, keypoints, descriptors);
		if(task == string("sift"))   run_sifthash(image, METHOD_SIFT, keypoints, descriptors);

		std::cerr << "computed descriptors: " << keypoints.size() << std::endl;
    cv::Mat image_display = image;
    cv::cvtColor(image_display, image_display, CV_GRAY2RGB);
		for (const cv::KeyPoint& keypoint: keypoints) {
		  cv::circle(image_display, keypoint.pt, keypoint.size, cv::Scalar(0, 255, 0), 1);
		}
		cv::imshow("LDAHash", image_display);
		cv::waitKey(0);
  }
}
