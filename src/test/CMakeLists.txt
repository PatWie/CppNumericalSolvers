# compile test
add_executable(verify verify.cpp )
target_link_libraries(verify gtest gtest_main pthread)
install (TARGETS verify DESTINATION bin)
add_test(NAME  verify 
	COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/verify
    )

