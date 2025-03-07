cmake_minimum_required(VERSION 3.12)
project(CppNumericalSolvers VERSION 1.0.0 LANGUAGES CXX)

add_library(CppNumericalSolvers INTERFACE)

target_include_directories(CppNumericalSolvers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(CppNumericalSolvers INTERFACE cxx_std_11)

