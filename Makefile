NVCC = /usr/local/cuda-9.0/bin/nvcc
CC =  g++
CXXFLAGS = -std=c++11
# Libraries for tigergpu
CPPFLAGS = -I/usr/local/cuda-9.0/include/ -I/home/ncaplar/CodeGpu/software/Random123-1.09/include -I/home/ncaplar/CodeGpu/tclap/include
LDFLAGS = -L/usr/local/cuda-9.0/lib64 -L/home/ncaplar/CodeGpu/software/lib -L/home/ncaplar/.conda/envs/idp2_schwimmbad/lib -L/home/ncaplar/.conda/envs/cuda_env/lib -lmpi -lcufft -lcuda -lcudart -ldtcmp

all:
	$(NVCC) -ccbin=g++ $(CXXFLAGS) -c main_cuFFT.cu $(CPPFLAGS) -arch=sm_60
	$(NVCC) -ccbin=g++ $(CXXFLAGS) main_cuFFT.o -o main_cuFFT $(CPPFLAGS) $(LDFLAGS) -arch=sm_60

