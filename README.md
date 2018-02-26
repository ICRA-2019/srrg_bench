# SRRG Benchmark Utilities
Collection of ground truth computation and performance evaluation utilities for SRRG packages

Affiliated packages: <br>
- [srrg_hbst](https://gitlab.com/srrg-software/srrg_hbst) - HBST: Hamming Binary Search Tree Header-only library <br>

Affiliated datasets: <br>
- [KITTI Visual Odometry / SLAM Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [The EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [mit stata center data set](http://projects.csail.mit.edu/stata/downloads.php)
- [Indoor Level 7 S-Block Dataset](https://wiki.qut.edu.au/display/cyphy/Indoor+Level+7+S-Block+Dataset)
- [The Málaga Stereo and Laser Urban Data Set](https://www.mrpt.org/MalagaUrbanDataset)
- [UQ St. Lucia Stereo Vehicular Dataset](http://asrl.utias.utoronto.ca/~mdw/uqstluciadataset.html)
- [Oxford Robotcar Dataset](http://robotcar-dataset.robots.ox.ac.uk/)
- [Nordlandsbanen](https://nrkbeta.no/2013/01/15/nordlandsbanen-minute-by-minute-season-by-season/)

Supported environments: <br>
- Ubuntu 14.04 with gcc/g++-5 or higher
- Ubuntu 16.04 with gcc/g++-5 or higher

### SRRG HBST
Image Retrieval (Closure) ground truth computation examples: <br>
- KITTI sequence 06:

    rosrun srrg_bench compute_closure_ground_truth -mode kitti -images 06.txt.d/ -poses 06_gt.txt

- Málaga extract 10:

    rosrun srrg_bench compute_closure_ground_truth -mode malaga -images Images/ -poses malaga-urban-dataset-extract-10_all-sensors_GPS.txt -use-gui
    
- St. Lucia:

    rosrun srrg_bench compute_closure_ground_truth -mode lucia -images 101215_153851_MultiCamera0/ -poses 101215_153851_Ins0.log -timestamps 101215_153851_MultiCamera0.log -ug
    
The command line argument `-use-gui` or `-ug` can be optionally appended to launch a viewer, <br>
displaying the obtained ground truth on the trajectory.

Brute-force filtered Descriptor Matching for the obtained Image Retrieval ground truth is triggered, <br>
by adding `-compute-confusion` or `-cc`. <br>
The brute-force filtering naturally requires an extensive amount of time for completion. <br>
A set of computed ground truth files can be found in the [srrg_hbst wiki](https://gitlab.com/srrg-software/srrg_hbst/wikis/home)

Geometric Verification on top of these filtered matches is performed <br>
when `-geometric-verification <camera_calibration>` or `-gv <camera_calibration>` is set. <br>
Note that the verification requires the camera calibration (e.g. `calib.txt` for KITTI) as additional parameter.

---
Benchmarks: <br>
- KITTI sequence 00 for HBST on pure trajectory ground truth:

    rosrun srrg_bench benchmark -mode kitti -images 00.txt.d/ -poses 00_gt.txt -method hbst
    
- KITTI sequence 00 for HBST with a generated ground truth file (filtered):

    rosrun srrg_bench benchmark -mode kitti -images 00.txt.d/ -poses 00_gt.txt -method hbst -closures <bf-filtered-closures>

- KITTI sequence 00 for FLANNLSH on pure trajectory ground truth:
    
    rosrun srrg_bench benchmark -mode kitti -images 00.txt.d/ -poses 00_gt.txt -method flannlsh

- KITTI sequence 00 for DBoW2 on pure trajectory ground truth:

    rosrun srrg_bench benchmark -mode kitti -images 00.txt.d/ -poses 00_gt.txt -method dbow2 -voc <vocabulary>
    
- Málaga extract 05 for HBST on pure trajectory ground truth with extracted keypoint display:

    rosrun srrg_bench benchmark -mode malaga -images Images/ -poses malaga-urban-dataset-extract-05_all-sensors_GPS.txt -method hbst -ug
    
- St. Lucia for DBoW2 with image Score Only on pure trajectory ground truth:

    rosrun srrg_bench benchmark -mode dbow2 -images 101215_153851_MultiCamera0/ -poses 101215_153851_Ins0.log -timestamps 101215_153851_MultiCamera0.log -so

---
Additional files: <br>
- DBoW2 BRIEF vocabulary:  (DLoopDetector)
- DBoW2 ORB vocabulary:  (ORB-SLAM2)
- Image Retrieval (Closure) ground truth files: [wiki of srrg_hbst](https://gitlab.com/srrg-software/srrg_hbst/wikis/home)
