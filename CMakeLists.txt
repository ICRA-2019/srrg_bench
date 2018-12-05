cmake_minimum_required(VERSION 2.8.3)
project(srrg_bench)

#ds determine build type, default build type Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()
message("${PROJECT_NAME}|build type: ${CMAKE_BUILD_TYPE}")

#ds flags for release build
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -pedantic -Ofast -DNDEBUG")
  message("${PROJECT_NAME}|adding flags for '${CMAKE_BUILD_TYPE}': '--std=c++11 -Wall -pedantic -Ofast -DNDEBUG'")
  
#ds flags for other build(s) (e.g. debug)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -pedantic -O0 -g -fstack-check")
  message("${PROJECT_NAME}|adding flags for '${CMAKE_BUILD_TYPE}': '--std=c++11 -Wall -pedantic -O0 -g -fstack-check'")
endif()

#ds specify target binary descriptor BIT size
add_definitions(-DDESCRIPTOR_SIZE_BITS=256)

#ds augmentation configuration
add_definitions(-DAUGMENTATION_WEIGHT=0)
add_definitions(-DAUGMENTATION_SIZE_BITS=0)

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
  message("${PROJECT_NAME}|found Eigen3: '${EIGEN3_INCLUDE_DIR}' ${EIGEN3_VERSION}")
  
  #ds enable hbst wrapping
  add_definitions(-DSRRG_HBST_HAS_EIGEN)
  include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
endif()
if(OpenCV_FOUND)
  message("${PROJECT_NAME}|found OpenCV: '${OpenCV_DIR}' ${OpenCV_VERSION}")
  set(SRRG_BENCH_BUILD_OPENCV true)
  
  #ds check for contrib library
  string(FIND "${OpenCV_LIBS}" "xfeatures2d" FOUND_OPENCV_CONTRIB)
  if(NOT ${FOUND_OPENCV_CONTRIB} EQUAL -1)
  	message("${PROJECT_NAME}|found xfeatures2d library, building contributed OpenCV components")
  	  add_definitions(-DSRRG_HBST_HAS_OPENCV_CONTRIB)
  endif()
  
  #ds enable hbst wrapping
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
  include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
endif()
if(srrg_hbst_FOUND)
  message("${PROJECT_NAME}|found srrg_hbst: '${srrg_hbst_DIR}' ${srrg_hbst_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_HBST)
  include_directories(${srrg_hbst_INCLUDE_DIRS})
  add_definitions(-DSRRG_MERGE_DESCRIPTORS) #ds enable descriptor merging
endif()
if(DBoW2_FOUND AND DLib_FOUND)
  message("${PROJECT_NAME}|found DBoW2: '${DBoW2_DIR}' ${DBoW2_VERSION}")
  message("${PROJECT_NAME}|found DLib: '${DLib_DIR}' ${DLib_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_DBOW2)
  include_directories(SYSTEM ${DLIB_INCLUDE_DIR})
  include_directories(SYSTEM ${DBOW2_INCLUDE_DIR})
endif()
if(FLANN_FOUND)
  message("${PROJECT_NAME}|found FLANN: '${FLANN_DIR}' ${FLANN_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_FLANNHC)
  include_directories(SYSTEM ${FLANN_INCLUDE_DIRS})
endif()
if(obindex2_FOUND)
  message("${PROJECT_NAME}|found OBIndex2: '${obindex2_DIR}' ${obindex2_VERSION}")
  add_definitions(-DSRRG_BENCH_BUILD_IBOW)
  include_directories(SYSTEM ${obindex2_INCLUDE_DIRS})
endif()
if(srrg_core_viewers_FOUND AND QGLVIEWER_FOUND)
  message("${PROJECT_NAME}|found required visualization libraries, building viewers")
  set(SRRG_BENCH_BUILD_VIEWERS true)
  add_definitions(-DSRRG_BENCH_BUILD_VIEWERS)
endif()

#ds environment path variables have to be defined manually
if(NOT "$ENV{SEGNET_PATH}" STREQUAL "" AND 
   NOT "$ENV{CUDA_PATH}" STREQUAL "" AND
   EXISTS "$ENV{SEGNET_PATH}" AND
   EXISTS "$ENV{CUDA_PATH}")

  message("${PROJECT_NAME}|found SegNet: '$ENV{SEGNET_PATH}'") 
  include_directories(SYSTEM "$ENV{SEGNET_PATH}/include")
  include_directories(SYSTEM "$ENV{SEGNET_PATH}/build/include")
  link_directories("$ENV{SEGNET_PATH}/build/lib")

  message("${PROJECT_NAME}|found CUDA: '$ENV{CUDA_PATH}'") 
  include_directories(SYSTEM "$ENV{CUDA_PATH}")
  set(SRRG_BENCH_BUILD_SEGNET true)
  add_definitions(-DSRRG_BENCH_BUILD_SEGNET)
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
message("${PROJECT_NAME}|building LDAHash descriptor")
add_subdirectory(src/thirdparty/ldahash)
message("${PROJECT_NAME}|building BOLD descriptor")
add_subdirectory(src/thirdparty/bold)

#ds add benchmark source folders
add_subdirectory(src/hbst)
add_subdirectory(src/bda)
