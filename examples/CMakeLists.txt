# Examles
set(CPP_FILES rosenbrock_function)

foreach(src_file ${CPP_FILES})
  add_executable(${src_file} ${src_file}.cpp)
  target_link_libraries(${src_file} PRIVATE ${LIB_FMT})
endforeach(src_file)
