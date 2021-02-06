# add_compile_options(-std=c++11)
add_compile_options(-g)
add_compile_options(-O0)

add_executable(test test.cc)
target_link_libraries(test
  ${OpenCV_LIBS}
)

add_executable(test_undistort test_undistort.cc)
target_link_libraries(test_undistort
  ${OpenCV_LIBS}
)