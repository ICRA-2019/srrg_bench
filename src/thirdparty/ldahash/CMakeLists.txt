#***************************************************************************
#*   Copyright (C) 2010 by Christoph Strecha   *
#*   christoph.strecha@epfl.ch   *
#***************************************************************************/
#ds All code in this folder has been obtained from: https://cvlab.epfl.ch/research/detect/ldahash
#ds And is courtesy of christoph.strecha@epfl.ch

#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds build LDAHash executables and library
add_executable(run_ldahash ldahash.cpp sift.cpp run_ldahash.cpp hammatch.cpp)
target_link_libraries(run_ldahash ${OpenCV_LIBS})

#ds build LDAHash evaluation tool
add_executable(evaluate_ldahash eval_descriptor.cpp)
target_link_libraries(evaluate_ldahash ${OpenCV_LIBS})

#ds LDAHash library
add_library(srrg_ldahash_descriptor_library ldahash.cpp sift.cpp hammatch.cpp)
target_link_libraries(srrg_ldahash_descriptor_library ${OpenCV_LIBS})
