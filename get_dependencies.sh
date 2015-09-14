# eigen library
wget -c http://bitbucket.org/eigen/eigen/get/3.2.2.tar.bz2 -O eigen-3.2.2.tar.bz2
bunzip2 eigen-3.2.2.tar.bz2
tar xvf eigen-3.2.2.tar
mv eigen-eigen-* eigen 
rm -Rf eigen-3.2.2.tar
rm -Rf eigen-3.2.2.tar.bz2