# BDA (Binary Descriptor Augmentation)
Adjusting the descriptor size (bits)
- For performance reasons the selected descriptor byte size has to be specified at compile time
- The descriptor byte size has to be adjusted when a descriptor of size other than 32 bytes (default) is chosen (e.g. BRISK-512 which is 64 bytes)
- The descriptor byte size can be set with the CMake variable `SRRG_BENCH_DESCRIPTOR_SIZE_BYTES` in the root `CMakeLists.txt` (line 17)

---
Image Retrieval benchmark examples
- [ZuBuD](http://www.vision.ee.ethz.ch/en/datasets) with [Brute-Force (BF)](https://docs.opencv.org/3.1.0/d3/da1/classcv_1_1BFMatcher.html) matching and BRIEF descriptors (default):

	    ./benchmark_map -mode zubud -images-query test/ -images-reference train/ -closures zubud_groundtruth.txt -method bf

- [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) with [FLANN-LSH](https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html) matching and ORB descriptors:

	    ./benchmark_map -mode oxford -images-query test/ -images-reference train/ -closures zubud_groundtruth.txt -method flannlsh -descriptor orb

- [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) with [Bag-of-Features (BOF)](https://github.com/dorian3d/DBoW2) matching and BRISK descriptors:

	    ./benchmark_map -mode oxford -images-query test/ -images-reference train/ -closures zubud_groundtruth.txt -method bof -descriptor brisk

- [Holidays](http://lear.inrialpes.fr/~jegou/data.php) with [HBST](https://gitlab.com/srrg-software/srrg_hbst) matching and A-KAZE descriptors:

	    ./benchmark_map -mode oxford -images-query test/ -images-reference train/ -closures zubud_groundtruth.txt -method hbst -descriptor akaze

- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with [HBST](https://gitlab.com/srrg-software/srrg_hbst) matching and FREAK descriptors:

	    ./benchmark_map -mode kitti -images image_0 -poses 00.txt -method hbst -descriptor freak

### It doesn't work? ###
[Open an issue](https://gitlab.com/srrg-software/srrg_bench/issues) or contact the maintainer (see package.xml)
