#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds check if we have visualization
if(SRRG_BENCH_BUILD_VIEWERS)

  #ds specify additional locations of header files
  include_directories(
    ${QGLVIEWER_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${SRRG_QT_INCLUDE_DIRS}
  )

  #ds help the catkin tool on 16.04 (cmake seems unable to find single libraries, although catkin claims the link_directories call is not required)
  #ds in order to avoid linking against the catkin_LIBRARIES bulk everytime enable this so one can select single libraries
  link_directories(${catkin_LIBRARY_DIRS})
  
  #ds resolve visualization library
  add_subdirectory(visualization)

  #ds build generic benchmark binary - with viewer
  add_executable(benchmark benchmark.cpp)
  target_link_libraries(benchmark srrg_bench_matchers_library srrg_bench_closure_viewer_library)

  #ds ground truth computation - with viewer
  add_executable(compute_closure_ground_truth compute_closure_ground_truth.cpp)
  target_link_libraries(compute_closure_ground_truth srrg_bench_matchers_library srrg_bench_closure_viewer_library)
  
  #ds compile live viewer
  add_executable(live_viewer live_viewer.cpp)
  target_link_libraries(live_viewer srrg_bench_loop_closure_evaluator_library srrg_bench_live_viewer_library pthread)

else()

  #ds build generic benchmark binary - without viewer
  add_executable(benchmark benchmark.cpp)
  target_link_libraries(benchmark srrg_bench_matchers_library)
  
  #ds ground truth computation - without viewer
  add_executable(compute_closure_ground_truth compute_closure_ground_truth.cpp)
  target_link_libraries(compute_closure_ground_truth srrg_bench_matchers_library)

endif()

#ds descriptor bit analysis for HBST
if(srrg_hbst_FOUND)
  add_executable(analyze_completeness_leafs analyze_completeness_leafs.cpp)
  target_link_libraries(analyze_completeness_leafs srrg_bench_loop_closure_evaluator_library)
  
  add_executable(analyze_completeness_hbst analyze_completeness_hbst.cpp)
  target_link_libraries(analyze_completeness_hbst srrg_bench_loop_closure_evaluator_library)
  
  add_executable(analyze_completeness_monte_carlo analyze_completeness_monte_carlo.cpp)
  target_compile_options(analyze_completeness_monte_carlo PUBLIC -fopenmp)
  target_link_libraries(analyze_completeness_monte_carlo srrg_bench_loop_closure_evaluator_library -fopenmp)
endif()
