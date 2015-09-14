# borrow from http://stackoverflow.com/questions/23083024/building-matlab-mex-files-with-cmake-for-g2o-library
set(Matlab_mex  /home/wieschol/apps/MATLAB/R2014a/bin/mex)

# find_package(MatlabMex)
# if(MATLABMEX_FOUND)
#   set(HAVE_MATLAB TRUE)
# endif()

set (CMAKE_CXX_COMPILER ${Matlab_mex})
set (CMAKE_C_COMPILER   ${Matlab_mex})

SET( CMAKE_SHARED_LIBRARY_SUFFIX .mexa64 )  # set suffix to .mexa64
SET( CMAKE_SHARED_LIBRARY_PREFIX )      # remove the "lib" prefix

SET( CMAKE_CXX_FLAGS "-cxx -largeArrayDims CXXFLAGS='${CXXFLAGS} -std=c++11' COPTIMFLAGS='$COPTIMFLAGS -O2'" )
SET( CMAKE_SHARED_LIBRARY_CXX_FLAGS )       # remove the -fPIC option. mex does not  accept the "-fPIC" option
SET( CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <FLAGS> -outdir <OBJECT_DIR> -c <SOURCE>; mv <OBJECT_DIR>/$$(basename <SOURCE> .cxx ).o <OBJECT>")

# #Variables controlling the linking-phase
SET( CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS )    # remove -shared options. mex does not accept the "-shared" option

SET(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")


SET( CMAKE_INSTALL_RPATH "\$ORIGIN" )
MESSAGE( STATUS "mex.cmake is loaded.\n" )

ADD_LIBRARY( cns SHARED cns.cxx )
