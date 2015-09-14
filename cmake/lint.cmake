set(CMAKE_SOURCE_DIR ..)
set(LINT_COMMAND ${CMAKE_SOURCE_DIR}/lint.py)
set(SRC_FILE_EXTENSIONS h hpp hu c cpp cu cc)
set(LINT_DIRS include/cns src/examples)

cmake_policy(SET CMP0009 NEW)  # suppress cmake warning

# find all files of interest
foreach(ext ${SRC_FILE_EXTENSIONS})
    foreach(dir ${LINT_DIRS})
        file(GLOB_RECURSE FOUND_FILES ${CMAKE_SOURCE_DIR}/${dir}/*.${ext})
        set(LINT_SOURCES ${LINT_SOURCES} ${FOUND_FILES})
    endforeach()
endforeach()

execute_process(
    COMMAND ${LINT_COMMAND} ${LINT_SOURCES}
    RESULT_VARIABLE LINT_NUM_ERRORS
    OUTPUT_VARIABLE LINT_RESULT
)

if(LINT_NUM_ERRORS GREATER 0)
    message(STATUS ${LINT_RESULT})
    if(LINT_NUM_ERRORS GREATER 1)
        message(FATAL_ERROR "Linter found ${LINT_NUM_ERRORS} errors!")
    else()
        message(FATAL_ERROR "Linter found ${LINT_NUM_ERRORS} error!")
    endif()
else()
    message(STATUS "Linter did not find any errors!")
endif()
