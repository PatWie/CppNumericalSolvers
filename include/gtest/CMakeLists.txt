add_library(gtest SHARED gtest.h gtest-all.cpp)

add_library(gtest_main gtest_main.cc)
target_link_libraries(gtest_main gtest)