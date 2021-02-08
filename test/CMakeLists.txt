# add_compile_options(-g)
# add_compile_options(-O0)

add_executable(test_initializer 
  test_initializer.cc
  ../frame.cc
  ../initializer.cc
)
target_link_libraries(test_initializer
  ${OpenCV_LIBS}
)

add_executable(test_undistort test_undistort.cc)
target_link_libraries(test_undistort
  ${OpenCV_LIBS}
)