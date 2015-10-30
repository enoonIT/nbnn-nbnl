
CXX = g++ -std=c++0x -O3 -msse -msse2 -fopenmp -ffast-math -DNDEBUG
#CXX = g++ -g
#CUDA_CXX = /idiap/temp/ikuzbor/local/bin/g++
CUDA_CXX = g++
CXX_OPTIONS = -O3 -msse -msse2 -fopenmp -ffast-math -DNDEBUG
#CXX_OPTIONS = -g
NVCC_OPTIONS=-Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant --diag_suppress=code_is_unreachable --diag_suppress=unrecognized_pragma --diag_suppress=unrecognized_gcc_pragma"
INCLUDE = -I. -IEigen

cu_smooth_ml3_ctypes: cu_smooth_ml3_ctypes.o CuSmoothML3.o CuMat.o
	nvcc cu_smooth_ml3_ctypes.o CuSmoothML3.o CuMat.o -o lib_cu_smooth_ml3_ctypes.so --shared -lcublas -lcuda -lcudart --ptxas-options=-v $(NVCC_OPTIONS) --compiler-options '$(CXX_OPTIONS)'

cu_smooth_ml3_ctypes.o: cu_smooth_ml3_ctypes.cu
	nvcc -c cu_smooth_ml3_ctypes.cu -o cu_smooth_ml3_ctypes.o -ccbin $(CUDA_CXX) $(INCLUDE) $(NVCC_OPTIONS) --compiler-options '$(CXX_OPTIONS) -fPIC'

CuMat.o: CuMat.cu
	nvcc -c CuMat.cu -o CuMat.o -ccbin $(CUDA_CXX) $(INCLUDE) $(NVCC_OPTIONS) --compiler-options '$(CXX_OPTIONS) -fPIC'

CuSmoothML3.o: CuSmoothML3.cu
	nvcc -c CuSmoothML3.cu -o CuSmoothML3.o -ccbin $(CUDA_CXX) $(INCLUDE) $(NVCC_OPTIONS) --compiler-options '$(CXX_OPTIONS) -fPIC'


mat_save_load_test: CuMat.o mat_save_load_test.cu
	nvcc mat_save_load_test.cu CuMat.o -o mat_save_load_test -lcublas -ccbin $(CUDA_CXX) $(INCLUDE) $(NVCC_OPTIONS)


clean:
	rm *.o; rm *.so; rm *pyc; rm *~


