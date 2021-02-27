#!/bin/bash

sudo apt-get install libgsl-dev
cd /home/runner/work/SOLikeT/SOLikeT/
wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz
tar -zxvf gsl-2.6.tar.gz
cd gsl-2.6
./configure --prefix=/home/runner/work/SOLikeT/SOLikeT/gsl-2.6
make
make install
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/runner/work/SOLikeT/SOLikeT/gsl-2.6/lib/
export LD_LIBRARY_PATH
dpkg -L libgsl-dev
git clone https://github.com/borisbolliet/class_sz.git
cd class_sz
make -j4

# at this point the make file leaves you in the python dir
cd ..
