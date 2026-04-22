load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")

def build_example(name, visibility=None):
  cc_binary(
    name = name,
    srcs = ["src/examples/"+name+".cc"],
    copts = ["-std=c++17", "-Wall", "-Wextra"],
    deps = [
        "//include:cppoptlib",
        "@eigen//:eigen",
    ]
  )

def build_test(name, visibility=None):
  cc_test(
    name = name,
    srcs = ["src/test/"+name+".cc"],
    copts = ["-Iexternal/gtest/include", "-std=c++17", " -Wall", "-Wextra"],
    deps = [
        "//include:cppoptlib",
        "@eigen//:eigen",
        "@googletest//:gtest_main",
    ]
  )
