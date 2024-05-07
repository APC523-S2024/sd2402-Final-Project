MACHINE = ADROIT

#==============================
# DELLA
#==============================
ifeq ($(MACHINE), ADROIT)
$(info	$(MACHINE))
#----- Directories ------------
IDIR = -I/usr/local/include/ -I/usr/include/ #-I/home/jaeukk/install/include # include directory
ODIR = ../obj/# object directory
EXEdir = ./EXC/# directory for executables

#	Executable
TARGET = $(EXEdir)soft_core_stealthy2.out #stitch.out

#----- Flags and libraries -----
GSLLIB = -L/usr/lib64 -lgsl -lgslcblas
# NLOPTLIB = -L/usr/lib64 -lnlopt 
HDF5LIB = -L/usr/lib64 -lhdf5 #a shot in the absolute pitch black and I think it worked
LIBS = -lm $(GSLLIB) $(HDF5LIB)
PREPRO = -DDELLA
endif

#==============================
#	Common flages
#==============================
CC = g++
DEBUG = -g
CFLAGS = -Wall -std=c++11 $(DEBUG) $(IDIR) -c -fopenmp -O3
LFLAGS = -Wall -std=c++11 $(DEBUG) $(LIBS) -fopenmp -O3

$(info	$(CFLAGS))
$(info	$(LFLAGS))

objs_core = GeometryVector.o RandomGenerator.o etc.o PeriodicCellList.o StructureFactor.o PairCorrelation.o
objs_pot = Potential.o StructureOptimization.o #MD_System.o
objs_main = main.o

DEPS = ../GeometryVector.h ../RandomGenerator.h ../etc.h ../PeriodicCellList.h ../StructureFactor.h ../PairCorrelation.h \
../Potential.h ../CollectiveCoordinatePotential.h ../StructureOptimization.h \
RepulsiveCCPotential.h #../MD_System.h

#  ../GenerateConfigs.h ../HardSphereFluids.h ../RandomSequentialAddition.h

OBJ_core = $(patsubst %.o, $(ODIR)%.o, $(objs_core))
OBJ_pot = $(patsubst %.o, $(ODIR)%.o, $(objs_pot))
OBJ_main = $(patsubst %.o, $(ODIR)%.o, $(objs_main))
OBJ = $(OBJ_core) $(OBJ_pot) $(OBJ_main)

$(info	$(OBJ))

SRC_core = $(patsubst %.o, ../%.cpp, $(objs_core))
SRC_pot = $(patsubst %.o, ../%.cpp, $(objs_pot))
SRC_main = $(patsubst %.o, %.cpp, $(objs_main))
#SRC_misc = $(patsubst %.o, ../%.cpp, $(objs_misc))

$(info	$(SRC_core))
$(info	$(SRC_pot))
$(info	$(SRC_main))

all: $(OBJ) $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(PREPRO) -o $@ $(OBJ) $(LFLAGS)

# compilations...
$(OBJ_main): $(ODIR)%.o: $(notdir %.cpp) $(DEPS)
	echo $@ $<
	$(CC) $(PREPRO) -o $@ $< $(CFLAGS) 

$(OBJ_core): $(ODIR)%.o: $(addprefix ../, %.cpp) $(DEPS)
	echo $@ $<  
	$(CC) $(PREPRO) -o $@ $< $(CFLAGS)  

$(OBJ_pot): $(ODIR)%.o: $(addprefix ../, %.cpp) $(DEPS)
	echo $@ $<  
	$(CC) $(PREPRO) -o $@ $< $(CFLAGS)  

clean:
	rm -f $(ODIR)*.o $(TARGET)

.PHONY: all
all:

