# compile test
add_executable(verify verify.cpp )
target_link_libraries(verify gtest gtest_main pthread)
install (TARGETS verify DESTINATION bin)
add_test(NAME  verify COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/verify)


add_executable(benchmark benchmark.cpp )
target_link_libraries(benchmark gtest gtest_main pthread)
install (TARGETS benchmark DESTINATION bin)
add_test(NAME  benchmark COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmark)

