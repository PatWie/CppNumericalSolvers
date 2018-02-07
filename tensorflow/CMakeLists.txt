cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{TensorFlow_GIT_REPO})
  message(STATUS "FOUND TENSORFLOW GIT-REPO in $ENV{TensorFlow_GIT_REPO}")
else()
    message(FATAL_ERROR "TensorFlow_GIT_REPO must be defined, see README.md")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt")
  include(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_config.txt)

  message(STATUS "TensorFlow_ABI: ${TensorFlow_ABI}")
  message(STATUS "TensorFlow_INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "You need to run \"python configure.py\" first")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )


project( TFCustomOp )


include_directories(SYSTEM "${TensorFlow_INCLUDE_DIRS}/external/nsync/public")
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIRS})
include_directories(SYSTEM ../include)

add_executable (solve solve.cpp)
TARGET_LINK_LIBRARIES(solve -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow_cc.so" -Wl,--no-whole-archive)
TARGET_LINK_LIBRARIES(solve -Wl,--allow-multiple-definition -Wl,--whole-archive "$ENV{TensorFlow_GIT_REPO}/bazel-bin/tensorflow/libtensorflow_framework.so" -Wl,--no-whole-archive)

