SET( EXAMPLE_FILES linearregression logisticregression rosenbrock simple nonnegls)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples )
foreach( currentfile ${EXAMPLE_FILES} )
	 add_executable( ${currentfile} "${currentfile}.cpp"  )
	 target_link_libraries (${currentfile} gtest gtest_main pthread)
endforeach( currentfile ${EXAMPLE_FILES} )
