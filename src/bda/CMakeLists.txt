#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds build benchmark binaries
add_executable(benchmark_map benchmark_map.cpp)
target_link_libraries(benchmark_map srrg_bench_matchers_library)

#ds build tests
if(SRRG_BENCH_BUILD_SEGNET)
  add_executable(test_segmentation_segnet test_segmentation_segnet.cpp)
  target_link_libraries(test_segmentation_segnet ${OpenCV_LIBS} caffe boost_system glog)
endif()
