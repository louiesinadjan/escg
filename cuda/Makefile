# Compiler
NVCC = nvcc
CXX = g++

# Compiler Flags
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -O3
NVCCFLAGS = -std=c++14 -Xcompiler "-Wall -Wextra -Werror"  -O3
# NVCCFLAGS = -std=c++17 -Xcompiler "-Wall -Wextra -Werror" -g -G

# Executable Name
TARGET = cuda-escg

# Source Files
SRC = cuda-escg.cu io.cu

# Default Rule
all: $(TARGET)

# Compile CUDA File
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

# Clean Build Files
clean:
	rm -f $(TARGET)