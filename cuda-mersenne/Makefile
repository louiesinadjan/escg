# Compiler
NVCC = nvcc
CXX = g++

# Compiler Flags
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror
NVCCFLAGS = -std=c++14 -Xcompiler "-Wall -Wextra -Werror" -g -G

# Executable Names
TARGETS = cuda-mersenne curand

# Source Files
SRC_CUDA_MERSENNE = cuda-mersenne.cu
SRC_CURAND = curand.cu

# Default Rule (Build All)
all: $(TARGETS)

# Compile Each CUDA File Separately
cuda-mersenne: $(SRC_CUDA_MERSENNE)
	$(NVCC) $(NVCCFLAGS) -o cuda-mersenne $(SRC_CUDA_MERSENNE)

curand: $(SRC_CURAND)
	$(NVCC) $(NVCCFLAGS) -o curand $(SRC_CURAND)

# Clean Build Files
clean:
	rm -f $(TARGETS)