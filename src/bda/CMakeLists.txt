#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds build benchmark binaries
add_executable(compute_ap compute_ap.cpp)
add_executable(benchmark_map benchmark_map.cpp)
target_link_libraries(benchmark_map srrg_bench_matchers_library)
