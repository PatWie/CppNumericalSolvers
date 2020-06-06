load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "eigen_archive",
  url = "https://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz",
  build_file = "@//:eigen.BUILD",
  strip_prefix = "eigen-eigen-5a0156e40feb",
  sha256 = "4286e8f1fabbd74f7ec6ef8ef31c9dbc6102b9248a8f8327b04f5b68da5b05e1"
)


http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "@//:gtest.BUILD",
    strip_prefix = "googletest-release-1.7.0",
)
