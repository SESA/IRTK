#dependencies
FLTK version 1.1.10
VTK 5 or 6
Boost, no earlier than version 1.48
GNU Scientific Library
CMAKE

#set up environment
export IRTK_DIR=/home/handong/build-irtk                                                                                                                                   
export PATH="$IRTK_DIR/bin:$PATH"  

#Edits
It requires gtest to be installed but I edited 
CMakeCache.txt and turned unit tests to OFF

#build and install
mkdir build
cd build
cmake ..


