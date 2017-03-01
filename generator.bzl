def build_example(name, visibility=None):
  native.cc_binary(
    name = name,
    srcs = ["src/examples/"+name+".cpp"],
    deps = [
        "//include/cppoptlib:cppoptlib",
        "@eigen_archive//:eigen3"
    ] 
  )

def build_test(name, visibility=None):
  native.cc_test(
    name = name,
    srcs = ["src/test/"+name+".cpp"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//include/cppoptlib:cppoptlib",
        "@eigen_archive//:eigen3",
        "@gtest//:main"
    ] 
  )
