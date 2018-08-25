#***************************************************************************
#*   Copyright (C) 2013 by Tomasz Trzcinski
#*   tomasz.trzcinski@epfl.ch
#***************************************************************************/
#ds All code in this folder has been obtained from: https://cvlab.epfl.ch/research/detect/binboost
#ds And is courtesy of tomasz.trzcinski@epfl.ch

#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds build BinBoost executables and library
include_directories("include")
add_executable(run_binboost src/BoostDesc.cpp src/Utils.cpp src/run_binboost.cpp)
target_link_libraries(run_binboost ${OpenCV_LIBS})
