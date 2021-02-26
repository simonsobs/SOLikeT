#!/bin/bash

export CONDA_BUILD_SYSROOT=/

rm -rf class_sz
git clone --depth=1000 https://github.com/borisbolliet/class_sz.git
cd class_sz

sed -i.bak -e 's/^CC/#CC/g' Makefile
sed -i.bak -e 's/^OPTFLAG =/OPTFLAG = ${CFLAGS} ${LDFLAGS}/g' Makefile
sed -i.bak -e 's/^#CCFLAG +=/CCFLAG +=/g' Makefile
sed -i.bak -e 's/^#CCFLAG =/CCFLAG =/g' Makefile

make -j4

# at this point the make file leaves you in the python dir
cd ..
