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

#ds specify target binary descriptor BIT size
add_definitions(-DDESCRIPTOR_SIZE_BITS=256)

#ds augmentation configuration
add_definitions(-DAUGMENTATION_WEIGHT=16)
add_definitions(-DAUGMENTATION_SIZE_BITS=12)

#ds locate external dependencies
find_package(catkin QUIET COMPONENTS srrg_cmake_modules srrg_core srrg_core_viewers srrg_gl_helpers srrg_hbst obindex2)
find_package(Eigen3 QUIET)
find_package(OpenCV QUIET)
find_package(DBoW2 QUIET)
find_package(DLib QUIET)
find_package(FLANN QUIET)

#ds viewer related
set(CMAKE_MODULE_PATH ${srrg_cmake_modules_INCLUDE_DIRS})
find_package(QGLViewer QUIET)

#ds evaluate configuration
if(EIGEN3_FOUND OR EIGEN3_INCLUDE_DIR)
  message("srrg_bench|found Eigen3: '${EIGEN3_INCLUDE_DIR}' ${EIGEN3_VERSION}")
  
  #ds enable hbst wrapping
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()
if(OpenCV_FOUND)
  message("srrg_bench|found OpenCV: '${OpenCV_DIR}' ${OpenCV_VERSION}")
  set(SRRG_BENCH_BUILD_OPENCV true)
  
  #ds enable hbst wrapping
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()
if(srrg_hbst_FOUND)
  message("srrg_bench|found srrg_hbst: '${srrg_hbst_DIR}' ${srrg_hbst_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_HBST)
  include_directories(${srrg_hbst_INCLUDE_DIRS})
endif()
if(DBoW2_FOUND AND DLib_FOUND)
  message("srrg_bench|found DBoW2: '${DBoW2_DIR}' ${DBoW2_VERSION}")
  message("srrg_bench|found DLib: '${DLib_DIR}' ${DLib_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_DBOW2)
  include_directories(${DLIB_INCLUDE_DIR})
  include_directories(${DBOW2_INCLUDE_DIR})
endif()
if(FLANN_FOUND)
  message("srrg_bench|found FLANN: '${FLANN_DIR}' ${FLANN_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_FLANNHC)
  include_directories(${FLANN_INCLUDE_DIRS})
endif()
if(obindex2_FOUND)
  message("srrg_bench|found OBIndex2: '${obindex2_DIR}' ${obindex2_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_IBOW)
  include_directories(${obindex2_INCLUDE_DIRS})
endif()
if(srrg_core_viewers_FOUND AND QGLVIEWER_FOUND)
  message("srrg_bench|found required visualization libraries, building viewers")
  set(SRRG_BENCH_BUILD_VIEWERS true)
  add_definitions(-DSRRG_BENCH_BUILD_VIEWERS)
endif()

#ds environment path variables have to be defined manually
if(NOT "$ENV{SEGNET_PATH}" STREQUAL "" AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
  message("srrg_bench|found SegNet: '$ENV{SEGNET_PATH}'") 
  include_directories("$ENV{SEGNET_PATH}/include")
  include_directories("$ENV{SEGNET_PATH}/build/include")
  link_directories("$ENV{SEGNET_PATH}/build/lib")

  message("srrg_bench|found CUDA: '$ENV{CUDA_PATH}'") 
  include_directories("$ENV{CUDA_PATH}")
  #set(SRRG_BENCH_BUILD_SEGNET true)
  #add_definitions(-DSRRG_BENCH_BUILD_SEGNET)
endif()

#ds add internal includes
include_directories(src)

#ds define catkin package
catkin_package(INCLUDE_DIRS src)

#ds dataset parsers, ground truth computation and parameter parsing
add_subdirectory(src/utilities)

#ds all benchmarked descriptor matchers
add_subdirectory(src/matchers)

#ds thirdparty software (e.g. descriptors)
message("srrg_bench|building LDAHash descriptor")
add_subdirectory(src/thirdparty/ldahash)
message("srrg_bench|building BOLD descriptor")
add_subdirectory(src/thirdparty/bold)

#ds add benchmark source folders
add_subdirectory(src/hbst)
add_subdirectory(src/bda)
