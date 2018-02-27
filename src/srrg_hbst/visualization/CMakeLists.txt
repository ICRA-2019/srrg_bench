add_library(srrg_bench_closure_viewer_library closure_viewer.cpp)
target_link_libraries(srrg_bench_closure_viewer_library 
  srrg_bench_loop_closure_evaluator_library
  srrg_core_viewers_library
  ${OpenCV_LIBS}
  ${OPENGL_gl_LIBRARY}
  ${OPENGL_glu_LIBRARY}
  ${QGLVIEWER_LIBRARY}
  ${SRRG_QT_LIBRARIES}
)

add_library(srrg_bench_live_viewer_library
            viewer_bonsai.cpp
            map_viewer.cpp
)
target_link_libraries(srrg_bench_live_viewer_library 
  srrg_core_viewers_library
  ${OpenCV_LIBS}
  ${OPENGL_gl_LIBRARY}
  ${OPENGL_glu_LIBRARY}
  ${QGLVIEWER_LIBRARY}
  ${SRRG_QT_LIBRARIES}
)
