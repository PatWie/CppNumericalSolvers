cmake_minimum_required(VERSION 3.1)
project(CppNumericalSolvers)

include(CppNumericalSolvers.config)

set( CMAKE_BUILD_TYPE Release)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_TEST_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# check existence of c++11 compiler
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
# set path to cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(LINKER_LIBS "")

# Google Testing Framework
add_subdirectory(gtest)
include_directories(gtest)

# Eigen3 Headerfiles
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})


add_subdirectory(src/examples)
add_subdirectory(src/test)


add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
