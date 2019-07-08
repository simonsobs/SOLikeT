include ${COSMOSIS_SRC_DIR}/config/compilers.mk
include Makefile.mine

ifeq ($(CFITSIO_INC)$(CFITSIO_LIB),) 
$(error To compile the ACTPOL likelihood you must activate the UPS package cfitsio or set the environment variables CFITSIO_INC and CFITSIO_LIB)
endif

ifeq ($(LAPACK_LINK),)
$(error To compile the ACTPOL likelihood you must activate the UPS package LAPACK or set the environment variable LAPACK_LIB)
endif

F90C    = ${FC}
INCS = -I$(CFITSIO_INC) 
LIBS = -L ${LAPACK_LINK} -L ${CFITSIO_LIB} -lcfitsio

FFLAGS+=-DACTPOL_DATA_DIR='"$(ACTPOL_DATA_DIR)/"'
FFLAGS+=  -ffree-line-length-none -cpp


ifeq ($(ACTPOL_DATA_DIR),)
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),distclean)
$(error To compile the ACT likelihood you must specify ACTPOL_DATA_DIR=/path/to/act/data in your Makefile.mine)
endif
endif
endif


#Makefile
CMBONLYLIBNAME = actpol
CMBONLYLIB=lib${CMBONLYLIBNAME}.a	

OBJS =  Options.o \
	actpol_vars.o \
	ACTPol_s2_like.o \

RM = rm -f

#------------------

#                       Rules.

PROGRAMS = test

all: $(PROGRAMS) $(CMBONLYLIB) interface

$(CMBONLYLIB): $(OBJS)
	ar r $@ $^

test: $(OBJS) test.o
	$(F90C) $(FFLAGS) -o $@ test.o $(OBJS) $(LDFLAGS) $(LIBS) 

%.o: %.f90
	$(F90C) $(FFLAGS) $(INCS) -c -o $@ $<

%.o: %.F90
	$(F90C) $(FFLAGS) $(INCS) -c -o $@ $<

clean:
	$(RM) *.o *.mod *.log *~ *.a test

distclean: clean
	$(RM) test

interface: $(CMBONLYLIB)
	$(F90C) $(FFLAGS) actpol_interface.f90 -shared -o actpol.so -L. -lactpol $(INCS) $(LDFLAGS) -lcosmosis_fortran $(LIBS)
