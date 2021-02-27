#!/bin/bash

sudo apt-get install libgsl-dev
dpkg -L libgsl-dev
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/share/miniconda/envs/test/lib/
export LD_LIBRARY_PATH
git clone https://github.com/borisbolliet/class_sz.git
cd class_sz
make -j4

# at this point the make file leaves you in the python dir
cd ..
