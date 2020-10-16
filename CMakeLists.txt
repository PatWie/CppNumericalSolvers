project(OptimiationSolvers)
cmake_minimum_required(VERSION 3.0)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(POSITION_INDEPENDENT_CODE ON)

set(CMAKE_INSTALL_PREFIX "$ENV{HOME}")
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EXTERNAL_DIR "${ROOT_DIR}/3p")
set(SRC_DIR "${ROOT_DIR}/src")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Read the git tags to determine the project version
include(GetGitVersion)
get_git_version(GIT_VERSION)

# Tell the user what versions we are using
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VERSION ${GIT_VERSION})
message("-- Version: ${VERSION}")

# The version of the libraries
set(GENERIC_LIB_VERSION ${VERSION})
string(SUBSTRING ${VERSION} 0 1 GENERIC_LIB_SOVERSION)

include(CheckCXXCompilerFlag)
include(AddCXXCompilerFlag)
include(CXXFeatureCheck)

# Enable other flags
add_cxx_compiler_flag(-std=c++17)
add_cxx_compiler_flag(-Wall)
add_cxx_compiler_flag(-pedantic)
add_cxx_compiler_flag(-pedantic-errors)
add_cxx_compiler_flag(-fstrict-aliasing)

if(HAVE_CXX_FLAG_FSTRICT_ALIASING)
  add_cxx_compiler_flag(-Wstrict-aliasing)
endif()

if(HAVE_WTHREAD_SAFETY)
  add_definitions(-DHAVE_WTHREAD_SAFETY)
  cxx_feature_check(THREAD_SAFETY_ATTRIBUTES)
endif()

# These flags are used to optimize the binary size.
# add_cxx_compiler_flag(-fno-exceptions)
add_cxx_compiler_flag(--fno-rtti)

# Customized flags
add_cxx_compiler_flag(-DFMT_HEADER_ONLY)
add_cxx_compiler_flag(-DCEREAL_RAPIDJSON_HAS_CXX11_RVALUE_REFS)

# Build using static libraries to reduce dependency.
add_cxx_compiler_flag(-flto)
add_cxx_compiler_flag(-static)

# Build with ASAN by default add_cxx_compiler_flag(-fsanitize=address)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include folders
include_directories("${EXTERNAL_DIR}/include")
include_directories("${EXTERNAL_DIR}/include/eigen3")
include_directories("include")

message("${EXTERNAL_DIR}/include/eigen3")

# Add libraries
find_library(LIB_FMT libfmt.a "${EXTERNAL_DIR}/lib" NO_DEFAULT_PATH)

# Add sub-directories
add_subdirectory(examples)
# add_subdirectory(unittets) add_subdirectory(benchmarks)

# Formating and linting all related files
file(GLOB_RECURSE SRCS_AND_HDRS *.cpp *.h)
list(FILTER SRCS_AND_HDRS EXCLUDE REGEX "3p/")
find_program(CLANG_FORMAT NAMES clang-format)
add_custom_command(
  OUTPUT .fmt_cpp_stamp
  DEPENDS ${SRCS_AND_HDRS}
  COMMAND
    ${CLANG_FORMAT}
    "-style={BasedOnStyle: google, DerivePointerAlignment: false}" -i
    ${SRCS_AND_HDRS}
  COMMAND touch .fmt_cpp_stamp
  COMMENT "Format C++ files with clang-format."
  VERBATIM)

file(GLOB CMAKE_FILES CMakeLists.txt)
find_program(CMAKE_FORMAT NAMES cmake-format)
if(${CMAKE_FORMAT} STREQUAL "CMAKE_FORMAT-NOTFOUND")
  add_custom_command(
    OUTPUT .fmt_cmake_stamp
    DEPENDS ${CMAKE_FILES}
    COMMAND touch .fmt_cmake_stamp
    COMMENT "Skip formatting CMake files."
    VERBATIM)
else()
  add_custom_command(
    OUTPUT .fmt_cmake_stamp
    DEPENDS ${CMAKE_FILES}
    COMMAND cmake-format -i ${CMAKE_FILES}
    COMMAND touch .fmt_cmake_stamp
    COMMENT "Format CMake files with cmake-format."
    VERBATIM)
endif()

file(GLOB SH_SRCS 3p/*.sh)
find_program(SHFMT NAMES shfmt)
if(${SHFMT} STREQUAL "SHFMT-NOTFOUND")
  add_custom_command(
    OUTPUT .fmt_bash_stamp
    DEPENDS ${SH_SRCS}
    COMMAND touch .fmt_bash_stamp
    COMMENT "Skip formatting Bash scripts."
    VERBATIM)
else()
  if(UNIX AND NOT APPLE)
    add_custom_command(
      OUTPUT .fmt_bash_stamp
      DEPENDS ${SH_SRCS}
      COMMAND ${SHFMT} -i 2 -ci -w ${SH_SRCS}
      COMMAND touch .fmt_bash_stamp
      COMMENT "Format Bash scripts with shfmt."
      VERBATIM)
  else()
    add_custom_command(
      OUTPUT .fmt_bash_stamp
      DEPENDS ${SH_SRCS}
      COMMAND touch .fmt_bash_stamp
      COMMENT "Skip formatting Bash scripts."
      VERBATIM)
  endif()
endif()

add_custom_target(
  fmt
  DEPENDS .fmt_bash_stamp .fmt_cpp_stamp .fmt_cmake_stamp
  COMMENT "Format Bash, C++, CMake, and Python files.")

add_custom_command(
  OUTPUT .lint_shellcheck_stamp
  DEPENDS ${SH_SRCS}
  COMMAND shellcheck ${SH_SRCS}
  COMMAND touch .lint_shellcheck_stamp
  COMMENT "Lint Bash files with shellcheck.")

add_custom_command(
  OUTPUT .lint_cmake_stamp
  DEPENDS ${CMAKE_FILES}
  COMMAND cmake-lint ${CMAKE_FILES}
  COMMAND touch .lint_cmake_stamp
  COMMENT "Lint CMake files with cmake-lint.")

add_custom_target(
  lint ALL
  DEPENDS .lint_shellcheck_stamp .lint_cmake_stamp
  COMMENT "Lint Bash and CMake files.")
