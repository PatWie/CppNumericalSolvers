disp('compiling ...');
mex -I./../eigen cns.cpp  CXXFLAGS="\$CXXFLAGS -std=c++11" COPTIMFLAGS="\$COPTIMFLAGS -O2" -output cns;
