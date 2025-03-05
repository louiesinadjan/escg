# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./metal-cpp -fobjc-arc
LDFLAGS = -framework Metal -framework Foundation -framework CoreGraphics

# Directories
SRC_DIR = .
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objs

# Source files
SERIAL_SRC = $(SRC_DIR)/serial-mersenne.cpp
METAL_SRC = $(SRC_DIR)/metal-mersenne.cpp

# Object files
SERIAL_OBJ = $(OBJ_DIR)/serial-mersenne.o
METAL_OBJ = $(OBJ_DIR)/metal-mersenne.o

# Executables
SERIAL_TARGET = $(BUILD_DIR)/serial-mersenne
METAL_TARGET = $(BUILD_DIR)/metal-mersenne

# Metal Shader
METAL_SHADER = mt19937.metal
METAL_LIB = $(BUILD_DIR)/mt19937.metallib

# Default target: Build both executables
all: $(SERIAL_TARGET) $(METAL_TARGET)

# Rule to create Serial Mersenne executable
$(SERIAL_TARGET): $(SERIAL_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SERIAL_OBJ)
	@echo "Built $(SERIAL_TARGET)"

# Rule to create Metal Mersenne executable
$(METAL_TARGET): $(METAL_OBJ) $(METAL_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(METAL_OBJ) $(LDFLAGS)
	@echo "Built $(METAL_TARGET)"

# Rule to compile Metal shader to metallib
$(METAL_LIB): $(METAL_SHADER)
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $(BUILD_DIR)/mt19937.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/mt19937.air -o $@
	@echo "Compiled Metal shader $< -> $@"

# Rule to compile .cpp files to .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Phony targets
.PHONY: all clean