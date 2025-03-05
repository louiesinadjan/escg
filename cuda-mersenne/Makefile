# Compiler
NVCC = nvcc
CXX = g++

# Compiler Flags
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror
NVCCFLAGS = -std=c++14 -Xcompiler "-Wall -Wextra -Werror" -g -G

# Executable Names
TARGETS = cuda-mersenne curand serial-mersenne

# Source Files
SRC_CUDA_MERSENNE = cuda-mersenne.cu
SRC_CURAND = curand.cu
SRC_SERIAL_MERSENNE = serial-mersenne.cpp

# Default Rule (Build All)
all: $(TARGETS)

# Compile Each CUDA File Separately
cuda-mersenne: $(SRC_CUDA_MERSENNE)
	$(NVCC) $(NVCCFLAGS) -o cuda-mersenne $(SRC_CUDA_MERSENNE)

curand: $(SRC_CURAND)
	$(NVCC) $(NVCCFLAGS) -o curand $(SRC_CURAND)

# Compile Serial Mersenne Twister (C++ Only)
serial-mersenne: $(SRC_SERIAL_MERSENNE)
	$(CXX) $(CXXFLAGS) -o serial-mersenne $(SRC_SERIAL_MERSENNE)

# Clean Build Files
clean:
	rm -f $(TARGETS)