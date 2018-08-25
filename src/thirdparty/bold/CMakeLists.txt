#ds All code in this folder has been obtained from: https://github.com/vbalnt/bold
#ds And is courtesy of balntas@robots.ox.ac.uk

#ds nothing to build here without eigen and opencv
if((NOT EIGEN3_FOUND) OR (NOT OpenCV_FOUND))
  return()
endif()

#ds build BOLD executables and library
add_executable(run_bold bold.cpp utils.cpp run_bold.cpp)
target_link_libraries(run_bold ${OpenCV_LIBS})

#ds BOLD library
add_library(srrg_bold_descriptor_library bold.cpp utils.cpp)
target_link_libraries(srrg_bold_descriptor_library ${OpenCV_LIBS})
