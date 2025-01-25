# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./metal-cpp -fobjc-arc
LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework Foundation -framework QuartzCore -framework CoreGraphics

# Directories
SRC_DIR = .
CONTROL_DIR = ./control
VIEW_DIR = ./view
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objs
CONTROL_OBJ_DIR = $(OBJ_DIR)/control
VIEW_OBJ_DIR = $(OBJ_DIR)/view

# Source files
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/timer.cpp 
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/timer.o 

# Metal Shader
METAL_SHADER = mt19937.metal
METAL_LIB = $(BUILD_DIR)/mt19937.metallib

# Output binary
TARGET = $(BUILD_DIR)/metal

# Default target
all: $(TARGET)

# Rule to create the output binary
$(TARGET): $(OBJS) $(METAL_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "Built $(TARGET)"

# Rule to compile Metal shader to metallib
$(METAL_LIB): $(METAL_SHADER)
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -c $< -o $(BUILD_DIR)/mt19937.air
	xcrun -sdk macosx metallib $(BUILD_DIR)/mt19937.air -o $@
	@echo "Compiled Metal shader $< -> $@"

# Rule to compile .cpp files to .o files
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(OBJ_DIR)/timer.o: $(SRC_DIR)/timer.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Phony targets
.PHONY: all clean