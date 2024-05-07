MACHINE = ADROIT

#==============================
# DELLA
#==============================
ifeq ($(MACHINE),ADROIT)
#----- Directories ------------
IDIR = /usr/local/include/# include directory
ODIR = ../obj/# object directory
EXEdir = ./EXC/# directory for executables

#	Executable
TARGET = $(EXEdir)manager.out

#----- Flags and libraries -----
GSLLIB = -lgsl -lgslcblas
LIBS = -lm $(GSLLIB)
endif

#==============================
#	Common flages
#==============================
CC = g++
DEBUG = -g
CFLAGS = -Wall -std=c++0x $(DEBUG) $(IDIR) -c -O3 -fopenmp
LFLAGS = -Wall -std=c++0x $(DEBUG) $(LIBS) -O3 -fopenmp

#==============================
#	Flags and libraries
#==============================
$(info	$(CFLAGS))
$(info	$(LFLAGS))


objs_core = GeometryVector.o PeriodicCellList.o RandomGenerator.o etc.o
objs_netw = main.o

DEPS = ../GeometryVector.h ../PeriodicCellList.h ../RandomGenerator.h ../etc.h

OBJ_core = $(patsubst %.o, $(ODIR)%.o, $(objs_core))
OBJ_netw = $(patsubst %.o, $(ODIR)%.o, $(objs_netw))  
OBJ = $(OBJ_core) $(OBJ_netw)
$(info  $(OBJ))

SRC_core = $(patsubst %.o, ../%.cpp, $(objs_core))
SRC_netw = $(patsubst %.o, %.cpp, $(objs_netw))

$(info  $(SRC_core))
$(info  $(SRC_netw))

all: $(OBJ) $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(LFLAGS)

# compilations...
$(OBJ_core): $(ODIR)%.o: $(addprefix ../, %.cpp) $(DEPS)
	echo $@ $<  
	$(CC) $(PREPROC) -o $@ $< $(CFLAGS)  

$(OBJ_netw): $(ODIR)%.o: $(notdir %.cpp) $(DEPS)
	echo $@ $<
	$(CC) $(PREPROC) -o $@ $< $(CFLAGS) 


clean:
	rm -f $(ODIR)*.o $(TARGET)

.PHONY: all
all:
