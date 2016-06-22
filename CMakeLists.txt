cmake_minimum_required(VERSION 2.8)
project(CppNumericalSolvers)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_TEST_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# check existence of c++11 compiler
message(STATUS "CHECK C++11 compatible compiler")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++0x")

# set path to cmake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(LINKER_LIBS "")

# Google Testing Framework
if (EXISTS "${PROJECT_SOURCE_DIR}/gtest/CMakeLists.txt")
  add_subdirectory(gtest)
else ()
  find_package(googletest REQUIRED)
endif ()
enable_testing()


# Eigen3 Headerfiles
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})


add_subdirectory(src/examples)
add_subdirectory(src/test)


add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
