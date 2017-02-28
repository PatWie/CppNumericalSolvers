#!/bin/bash
# eigen library
EIGEN_VERSION="3.2.10"
curl https://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz | tar xz
mv eigen-eigen-* eigen
