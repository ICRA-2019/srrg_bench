# SRRG Benchmark Utilities
Collection of ground truth computation and performance evaluation utilities for SRRG packages <br>
This project is best built using the [catkin command line tools](https://catkin-tools.readthedocs.io)

Dependencies: <br>
- [srrg_cmake_modules](https://gitlab.com/srrg-software/srrg_cmake_modules)
- [srrg_core](https://gitlab.com/srrg-software/srrg_core)
- [srrg_core_viewers](https://gitlab.com/srrg-software/srrg_core_viewers) (optional, for visualization)
- [srrg_gl_helpers](https://gitlab.com/srrg-software/srrg_gl_helpers) (optional, for visualization)

Affiliated packages (required if benchmarking is desired): <br>
- [srrg_hbst](https://gitlab.com/srrg-software/srrg_hbst) - HBST: Hamming Binary Search Tree Header-only library <br>

Generic SLAM datasets: <br>
- [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [The EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [mit stata center data set](http://projects.csail.mit.edu/stata/downloads.php)
- [Indoor Level 7 S-Block Dataset](https://wiki.qut.edu.au/display/cyphy/Indoor+Level+7+S-Block+Dataset)
- [The Málaga Stereo and Laser Urban Data Set](https://www.mrpt.org/MalagaUrbanDataset)
- [UQ St. Lucia Stereo Vehicular Dataset](http://asrl.utias.utoronto.ca/~mdw/uqstluciadataset.html)
- [Oxford Robotcar Dataset](http://robotcar-dataset.robots.ox.ac.uk/)
- [Nordlandsbanen](https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/)

Pure Image Retrieval (VPR) datasets:

- [ZuBuD](http://www.vision.ee.ethz.ch/en/datasets)
- [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)
- [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
- [Holidays](http://lear.inrialpes.fr/~jegou/data.php)

Supported environments: <br>
- Ubuntu 14.04 with gcc 5 or higher
- Ubuntu 16.04 with gcc 5 or higher
- Ubuntu 18.04 with gcc 7 or higher

Reference software (required if a comparison is desired): <br>
- DBoW2: https://github.com/schdomin/DBoW2 (fork from the official repository, added generic descriptor support)
- FLANN: https://github.com/mariusmuja/flann (used for Hierarchical Clustering Trees)
- iBoW: https://github.com/emiliofidalgo/obindex2

# Available benchmarks
- [Hamming Binary Search Tree (HBST)](https://gitlab.com/srrg-software/srrg_bench/tree/master/src/hbst)
- [Binary Descriptor Augmentation (BDA)](https://gitlab.com/srrg-software/srrg_bench/tree/master/src/bda)

### It doesn't work? ###
[Open an issue](https://gitlab.com/srrg-software/srrg_bench/issues) or contact the maintainer (see package.xml)
