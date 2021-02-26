#!/bin/bash

git clone https://github.com/borisbolliet/class_sz.git
cd class_sz
make -j4

# at this point the make file leaves you in the python dir
cd ..
