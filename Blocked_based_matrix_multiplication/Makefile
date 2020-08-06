.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lcudart
CC	:= nvcc
DEP	:=  
ifeq ($(def), )
DEF := 
else
DEF := -D$(def)
endif

NVCCFLAGS	:= -lineinfo -arch=sm_53 --ptxas-options=-v -g 

all:	const shared global hw1

const:	cmpe214_matMul_const.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_const cmpe214_matMul_const.o $(LIB)

shared:	cmpe214_matMul_shared.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_shared cmpe214_matMul_shared.o $(LIB)

global:	cmpe214_matMul_global.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_global cmpe214_matMul_global.o $(LIB)

hw1: cmpe214_matMul_hw1.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o Blocked_based_matrix_multiplication cmpe214_matMul_hw1.o $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@
	

clean:
	rm -f *.o matrixMul_*


