file(GLOB g2o_libs ../third_party/g2o/lib/libg2o_*.so)
file(GLOB dbow2_lib ../third_party/DBoW2/lib/libDBoW2.so)
message(STATUS "g2o_libs: ${g2o_libs}")

# undistort
add_executable(test_undistort test_undistort.cc)
target_link_libraries(test_undistort
  ${OpenCV_LIBS}
)

# undistort fisheye
add_executable(test_undistort_fisheye 
  test_undistort_fisheye.cc 
  ../camera_model.cc
  ../frame.cc
  ../utils.cc
)
target_link_libraries(test_undistort_fisheye
  ${OpenCV_LIBS}
  ${dbow2_lib}
  ${YAML_CPP_LIBRARIES}
)

# initialize map
add_executable(test_initialize_map
  test_initialize_map.cc
  ../frame.cc
  ../map.cc
  ../optimizer.cc
  ../utils.cc
  ../mappoint.cc
  ../camera_model.cc
  ../g2o_types_linear_motion.cc
)
target_link_libraries(test_initialize_map
  ${OpenCV_LIBS}
  ${g2o_libs}
  ${dbow2_lib}
  ${YAML_CPP_LIBRARIES}
)

# system
add_executable(test_system
  test_system.cc
  ../frame.cc
  ../map.cc
  ../system.cc
  ../optimizer.cc
  ../utils.cc
  ../mappoint.cc
  ../camera_model.cc
  ../g2o_types_linear_motion.cc
)
target_link_libraries(test_system
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  ${g2o_libs}
  ${dbow2_lib}
  pthread
)


# DBoW2
add_executable(test_DBoW2
  test_DBoW2.cc
  ../frame.cc
  ../utils.cc
  ../localization.cc
)
target_link_libraries(test_DBoW2
  ${OpenCV_LIBS}
  ${dbow2_lib}
  pthread
)

add_executable(test_trainingVoc
  test_trainingVoc.cc
  ../frame.cc
  ../utils.cc
)
target_link_libraries(test_trainingVoc
  ${OpenCV_LIBS}
  ${dbow2_lib}
  pthread
)
