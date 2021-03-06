#include "utils.h"
#include <iostream>

/* init the dataset: patches & ground truth  */
void init_dataset(dataset *A,const char *path)
{
  /* read the # of images */
  int numImgs=0;
  DIR *dir;
  struct dirent *ent;
  dir = opendir(path);
  if (dir != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      char *ext = strrchr(ent->d_name, '.');
      if (strcmp (ext,".bmp")==0)
  	{
  	  numImgs++;
  	}
    }
    closedir (dir);
  }
  std::cerr << "images: " << numImgs << std::endl;
  char s[100];
  /* loop through all images and read them */
  for (int i = 0; i < numImgs-1; i++) {
    int N = 32;
    sprintf(s, "%spatches%04d.bmp",path ,i);
    std::cerr << s << std::endl;
    cv::Mat largeImg = cv::imread(s, 0);
    for (int r = 0; r < largeImg.rows; r += N)
      for (int c = 0; c < largeImg.cols; c += N)
	{
	  cv::Mat tile = largeImg(cv::Range(r, cv::min(r + N, largeImg.rows)),
				  cv::Range(c, cv::min(c + N, largeImg.cols)));
	  A->patchesCV.push_back(tile);
    }
  }

  char gt_fname[55];
  sprintf(gt_fname, "%sm50_500000_500000_0.txt",path);

  /* read the gt file */
  FILE *in_file;
  in_file = fopen("info.txt", "rb");
  /* init the gt */
  A->gt =(int**) malloc(GT_SIZE * sizeof(int *));
  for (int i = 0; i < GT_SIZE; i++){
    A->gt[i] = (int*)malloc(7 * sizeof(int));
  }

  /* read the gt */
  for (int i = 0; i < GT_SIZE; i++) {
    for (int j = 0; j < 7 ; ++j){
      int fscan_res;
      fscan_res = fscanf(in_file, "%d", &A->gt[i][j]);
      if (!fscan_res)
  	{
  	  printf("Something went wrong. \n");
  	}
    }
  }
  fclose(in_file);
}
