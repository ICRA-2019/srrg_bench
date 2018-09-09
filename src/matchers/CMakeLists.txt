#ds HBST
if(srrg_hbst_FOUND)
  set(MATCHERS_SOURCES "hbst_matcher.cpp;${MATCHERS_SOURCES}")
endif()

#ds always build BF and FLANN-LSH: OpenCV
set(MATCHERS_SOURCES "bruteforce_matcher.cpp;flannlsh_matcher.cpp;${MATCHERS_SOURCES}")
set(MATCHERS_LIBRARIES "${OpenCV_LIBS};${MATCHERS_LIBRARIES}")

#ds DBoW2
if(DBoW2_FOUND AND DLib_FOUND)
  set(MATCHERS_SOURCES "bow_matcher.cpp;${MATCHERS_SOURCES}")
  set(MATCHERS_LIBRARIES "${DLib_LIBRARY};${DBoW2_LIBRARY};${MATCHERS_LIBRARIES}")
endif()

#ds FLANN-HC (native)
if(FLANN_FOUND)
  set(MATCHERS_SOURCES "flannhc_matcher.cpp;${MATCHERS_SOURCES}")
  
  #ds adjust FLANN dependencies (encountered problems on Ubuntu 16.04.3 LTS without lz4)
  set(MATCHERS_LIBRARIES "${FLANN_LIBRARIES};${MATCHERS_LIBRARIES};lz4")
endif()

#ds iBoW-LCD -> obindex
if(obindex2_FOUND)
  set(MATCHERS_SOURCES "ibow_matcher.cpp;${MATCHERS_SOURCES}")
  set(MATCHERS_LIBRARIES "gomp;${obindex2_LIBRARIES};${MATCHERS_LIBRARIES}")
endif()

#ds create target library
if(MATCHERS_SOURCES)
  add_library(srrg_bench_matchers_library ${MATCHERS_SOURCES})
  if(obindex2_FOUND)
    target_compile_options(srrg_bench_matchers_library PUBLIC -fopenmp)
  endif()
  target_link_libraries(srrg_bench_matchers_library 
    ${MATCHERS_LIBRARIES}
    srrg_bench_loop_closure_evaluator_library
  )
endif()
