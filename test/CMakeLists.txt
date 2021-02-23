# undistort
add_executable(test_undistort test_undistort.cc)
target_link_libraries(test_undistort
  ${OpenCV_LIBS}
)
# undistort
add_executable(test_undistort_fisheye 
  test_undistort_fisheye.cc 
  ../undistort.cc
)
target_link_libraries(test_undistort_fisheye
  ${OpenCV_LIBS}
)

# initialize map
add_executable(test_initialize_map
  test_initialize_map.cc
  ../frame.cc
  ../initializer.cc
  ../map.cc
  ../optimizer.cc
  ../utils.cc
  ../mappoint.cc
)

file(GLOB g2o_libs ../third_party/g2o/lib/libg2o_*.so)
message(STATUS "g2o_libs: ${g2o_libs}")
target_link_libraries(test_initialize_map
  ${OpenCV_LIBS}
  ${g2o_libs}
)

# system
add_executable(test_system
  test_system.cc
  ../frame.cc
  ../initializer.cc
  ../map.cc
  ../system.cc
  ../optimizer.cc
  ../utils.cc
  ../mappoint.cc
)

file(GLOB g2o_libs ../third_party/g2o/lib/libg2o_*.so)
message(STATUS "g2o_libs: ${g2o_libs}")
target_link_libraries(test_system
  ${OpenCV_LIBS}
  ${g2o_libs}
  pthread
)
