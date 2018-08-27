# BDA (Binary Descriptor Augmentation)
Adjusting the descriptor size (bits)
- For performance reasons the selected descriptor byte size has to be specified at compile time
- The descriptor byte size has to be adjusted when a descriptor of size other than 32 bytes (default) is chosen (e.g. BRISK-512 which is 64 bytes)
- The descriptor byte size can be set with the CMake variable `SRRG_BENCH_DESCRIPTOR_SIZE_BYTES` in the root `CMakeLists.txt` (line 17)

---
Image Retrieval benchmark examples
- [ZuBuD](http://www.vision.ee.ethz.ch/en/datasets) with [Brute-Force (BF)](https://docs.opencv.org/3.1.0/d3/da1/classcv_1_1BFMatcher.html) matching and BRIEF descriptors (default)
without augmentation:

	    ./benchmark_map -mode zubud -images-query zubud/1000city/qimage/ -images-reference zubud/png-ZuBuD/ -closures zubud_groundtruth.txt -method bf

- [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) with [FLANN-LSH](https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html) matching and ORB descriptors
with a 5x5 PA at weight 1:

	    ./benchmark_map -mode oxford -images-query oxford/ -images-reference oxford/ -method flannlsh -descriptor orb -position-augmentation 5 5 1

- [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) with [Bag-of-Features (BOF)](https://github.com/dorian3d/DBoW2) matching and BRISK descriptors
with a 5x5 PA at weight 2:

	    ./benchmark_map -mode paris -images-query paris/ -images-reference paris/ -method bof -descriptor brisk -position-augmentation 5 5 2

- [Holidays](http://lear.inrialpes.fr/~jegou/data.php) with [HBST](https://gitlab.com/srrg-software/srrg_hbst) matching and A-KAZE descriptors
with a 9x9 PA at weight 4:

	    ./benchmark_map -mode holidays -images-query holidays/jpg/ -closures holidays/eval_holidays/perfect_result.dat -method hbst -descriptor akaze -position-augmentation 9 9 4

- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with [HBST](https://gitlab.com/srrg-software/srrg_hbst) matching and FREAK descriptors
without augmentation:

	    ./benchmark_map -mode kitti -images kitti_00/image_0 -poses kitti_00/00.txt -method hbst -descriptor freak

Source images and ground truth files have been extracted from the compressed files provided online <br>
We extracted all files into a corresponding, separate dataset folder (e.g. ZuBuD in `zubud`, Oxford in `oxford`) <br>
We did not alter the file hierarchy, nor any of the file names except for [Holidays](http://lear.inrialpes.fr/~jegou/data.php) where we merged the directory `jpg(2)` into `jpg`

### It doesn't work? ###
[Open an issue](https://gitlab.com/srrg-software/srrg_bench/issues) or contact the maintainer (see package.xml)
