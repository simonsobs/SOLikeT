#!/bin/bash

export CONDA_BUILD_SYSROOT=/

rm -rf class_public
git clone --depth=1000 https://github.com/lesgourg/class_public.git
cd class_public

# if you are running on macos 11.6 and higher, uncomment the following line
sed -i.bak -e 's/gcc/gcc-9/g' Makefile

# if you are running on macos 10.15 and lower, uncomment the following 4 lines
#sed -i.bak -e 's/^CC/#CC/g' Makefile
#sed -i.bak -e 's/^OPTFLAG =/OPTFLAG = ${CFLAGS} ${LDFLAGS}/g' Makefile
#sed -i.bak -e 's/^#CCFLAG +=/CCFLAG +=/g' Makefile
#sed -i.bak -e 's/^#CCFLAG =/CCFLAG =/g' Makefile

make -j4

# at this point the make file leaves you in the python dir
cd ..
