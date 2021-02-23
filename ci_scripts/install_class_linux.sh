#!/bin/bash

git clone https://github.com/lesgourg/class_public.git
cd class_public
make -j4

# at this point the make file leaves you in the python dir
cd ..
