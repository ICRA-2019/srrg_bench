# HBST (Hamming Binary Search Tree)
Adjusting the descriptor size (bits)
- For performance reasons the selected descriptor byte size has to be specified at compile time
- The descriptor byte size has to be adjusted when a descriptor of size other than 32 bytes (default) is chosen (e.g. BRISK-512 which is 64 bytes)
- The descriptor byte size can be set with the CMake variable `SRRG_BENCH_DESCRIPTOR_SIZE_BYTES` in the root `CMakeLists.txt` (line 17)

---
Analytical tools
- Monte-Carlo BST sampling (random bit selection, for **all** nodes) with OpenMP support:

        rosrun srrg_bench analyze_completeness_monte_carlo -mode kitti -images image_0 -poses 00.txt -descriptor brief -depth 10 -samples 100 -threads 4

- Complete split evaluation in leafs (mean bit selection for nodes):

        rosrun srrg_bench analyze_completeness_leafs -mode kitti -images image_0 -poses 00.txt -descriptor brief -depth 10

- Mean split evaluation (proposed in HBST, balanced and incremental construction):

        rosrun srrg_bench analyze_completeness_hbst -mode kitti -images image_0 -poses 00.txt -descriptor brief -depth 10

---
Image Retrieval (Closure) ground truth computation examples
- KITTI sequence 06:

        rosrun srrg_bench compute_closure_ground_truth -mode kitti -images image_0 -poses 06.txt

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
Benchmarks
- KITTI sequence 00 for HBST on pure trajectory ground truth:
  
        rosrun srrg_bench benchmark -mode kitti -images image_0 -poses 00.txt -method hbst
    
- KITTI sequence 00 for HBST with a generated ground truth file (filtered):

        rosrun srrg_bench benchmark -mode kitti -images image_0 -poses 00.txt -method hbst -closures <bf-filtered-closures>

- KITTI sequence 00 for FLANNLSH on pure trajectory ground truth:
    
        rosrun srrg_bench benchmark -mode kitti -images image_0 -poses 00.txt -method flannlsh

- KITTI sequence 00 for DBoW2 on pure trajectory ground truth:

        rosrun srrg_bench benchmark -mode kitti -images image_0 -poses 00.txt -method dbow2 -voc <vocabulary>
    
- Málaga extract 05 for HBST on pure trajectory ground truth with extracted keypoint display:

        rosrun srrg_bench benchmark -mode malaga -images Images/ -poses malaga-urban-dataset-extract-05_all-sensors_GPS.txt -method hbst -ug
    
- St. Lucia for DBoW2 with image Score Only on pure trajectory ground truth:

        rosrun srrg_bench benchmark -mode dbow2 -images 101215_153851_MultiCamera0/ -poses 101215_153851_Ins0.log -timestamps 101215_153851_MultiCamera0.log -so

---
Additional files <br>
- DBoW2 BRIEF vocabulary: https://drive.google.com/open?id=1J5mQH96GA8zfsp0AYx2XdfrZIpAUwEhK (DLoopDetector)
- DBoW2 ORB vocabulary: https://drive.google.com/open?id=1Yh83ZfAH18m035y2PJiPeN0KUhDc8UD1 (ORB-SLAM2)
- Image Retrieval (Closure) ground truth files: [srrg_hbst wiki](https://gitlab.com/srrg-software/srrg_hbst/wikis/home)

### It doesn't work? ###
[Open an issue](https://gitlab.com/srrg-software/srrg_bench/issues) or contact the maintainer (see package.xml)
