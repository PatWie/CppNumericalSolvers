# Build project in release version
mkdir build
cd build
cmake -DEIGEN3_INCLUDE_DIR=eigen -DCMAKE_BUILD_TYPE=Release ..
make clean
make all
./bin/verify