# BDA (Binary Descriptor Augmentation)
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

### It doesn't work? ###
[Open an issue](https://gitlab.com/srrg-software/srrg_bench/issues) or contact the maintainer (see package.xml)
