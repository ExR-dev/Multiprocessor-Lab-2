
all: gaussjordan_sequential gaussjordan_parallel


oddeven: oddeven_sequential oddeven_parallel

gauss: gaussjordan_sequential gaussjordan_parallel


oddeven_sequential: oddevensort.cpp
	g++ -std=c++0x -O2 -o oddeven_sequential oddevensort.cpp

oddeven_parallel: oddevensortpar.cu
	nvcc -o oddeven_parallel oddevensortpar.cu
	

gaussjordan_sequential: gaussjordanseq.c
	gcc -O2 -o gaussjordan_sequential gaussjordanseq.c

gaussjordan_parallel: gaussjordanpar.cu
	nvcc -o gaussjordan_parallel gaussjordanpar.cu


clean:
	rm *.o *.a oddeven_sequential oddeven_parallel gaussjordan_sequential gaussjordan_parallel
