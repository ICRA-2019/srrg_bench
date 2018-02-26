cmake_minimum_required(VERSION 2.8.3)
project(srrg_bench)

#ds determine build type, default build type Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("srrg_bench|build type: ${CMAKE_BUILD_TYPE}")

#ds add c++11 support, mediocre optimization, all warnings
add_definitions(-Ofast -march=native -Wall -pedantic -std=c++11)

#ds debug flags
#add_definitions(-O0 -march=native -Wall -pedantic -std=c++11 -g)

#ds locate external dependencies
find_package(catkin QUIET COMPONENTS srrg_cmake_modules srrg_core_viewers srrg_gl_helpers obindex2)
find_package(Eigen3 QUIET)
find_package(OpenCV QUIET)
find_package(DBoW2 QUIET)
find_package(DLib QUIET)
find_package(FLANN QUIET)

#ds viewer related
set(CMAKE_MODULE_PATH ${srrg_cmake_modules_INCLUDE_DIRS})
find_package(QGLViewer QUIET)

#ds evaluate configuration
if(EIGEN3_FOUND)
  message("srrg_bench|found Eigen3: ${EIGEN3_INCLUDE_DIR}")
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
endif()
if(OpenCV_FOUND)
  message("srrg_bench|found OpenCV: ${OpenCV_DIR}")
  set(SRRG_BENCH_BUILD_OPENCV true)
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
endif()
if(DBoW2_FOUND AND DLib_FOUND)
  message("srrg_bench|found DBoW2: ${DBoW2_DIR}")
  message("srrg_bench|found DLib: ${DLib_DIR}")
  set(SRRG_BENCH_BUILD_DBOW2 true)
endif()
if(FLANN_FOUND)
  message("srrg_bench|found FLANN: ${FLANN_DIR}")
  set(SRRG_BENCH_BUILD_FLANN true)
endif()
if(obindex2_FOUND)
  message("srrg_bench|found OBIndex2: ${obindex2_DIR}")
  set(SRRG_BENCH_BUILD_IBOW true)
endif()
if(srrg_core_viewers_FOUND AND QGLVIEWER_FOUND)
  message("srrg_bench|found required visualization libraries, building viewers")
  set(SRRG_BENCH_BUILD_VIEWERS true)
endif()

