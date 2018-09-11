add_library(srrg_bench_loop_closure_evaluator_library loop_closure_evaluator.cpp command_line_parameters.cpp)

#ds with segnet we need to link against its fiendish libraries
if(SRRG_BENCH_BUILD_SEGNET)
target_link_libraries(srrg_bench_loop_closure_evaluator_library 
	srrg_bold_descriptor_library
	srrg_ldahash_descriptor_library
	${OpenCV_LIBS}
  caffe
  boost_system
  glog
)
else()
target_link_libraries(srrg_bench_loop_closure_evaluator_library 
  srrg_bold_descriptor_library
  srrg_ldahash_descriptor_library
  ${OpenCV_LIBS}
)
endif()
