#!/bin/bash

sudo apt-get install libgsl-dev
dpkg -L libgsl-dev
git clone https://github.com/borisbolliet/class_sz.git
cd class_sz
make -j4

# at this point the make file leaves you in the python dir
cd ..
