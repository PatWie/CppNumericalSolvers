mkdir build
cd build
CMAKE_CXX_COMPILER=clang++ CMAKE_BUILD_TYPE=Release cmake ..
make clean
make all
./bin/verify