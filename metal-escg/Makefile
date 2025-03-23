.DELETE_ON_ERROR:

# Compiler and flags
CXX = clang++
DEBUG ?= 0
CXXFLAGS = -std=c++17 -Wall -Wextra -I./metal-cpp -fobjc-arc

ifeq ($(DEBUG), 1)
	CXXFLAGS += -fsanitize=address -g
else
	CXXFLAGS += -O3 -march=native
endif

# Attempt to use --embed; if unavailable, fallback to normal --ldflags
EMBED_LDFLAGS = $(shell (python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags))

# Dynamically fetch other needed paths
PYTHON_INCLUDE  := $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
NUMPY_INCLUDE   := $(shell python3 -c "import numpy; print(numpy.get_include())")

# Additional include paths:
INCLUDE_FLAGS = -Imetal-escg/external/matplotlib-cpp \
                -I$(PYTHON_INCLUDE) \
                -I$(NUMPY_INCLUDE)

# Some systems require -undefined dynamic_lookup to load Python symbols at runtime on macOS
# We combine that with the embed or fallback flags.
# LDFLAGS = $(EMBED_LDFLAGS) -undefined dynamic_lookup -framework Metal -framework MetalKit -framework AppKit -framework Foundation -framework QuartzCore -framework CoreGraphics
LDFLAGS = $(EMBED_LDFLAGS) -undefined dynamic_lookup -framework Metal -framework Foundation -framework CoreGraphics

# Directories
SRC_DIR = .
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objs

# Source files
SRCS = $(SRC_DIR)/metal-escg.cpp $(SRC_DIR)/visualise.cpp $(SRC_DIR)/io.cpp
OBJS = $(OBJ_DIR)/metal-escg.o $(OBJ_DIR)/visualise.o $(OBJ_DIR)/io.o

# Metal Shaders
METAL_SHADERS = random.metal densities.metal step.metal
METAL_LIB = $(BUILD_DIR)/escg.metallib
METAL_AIR_FILES = $(patsubst %.metal,$(BUILD_DIR)/%.air,$(METAL_SHADERS))

# Output binary
TARGET = $(BUILD_DIR)/metal

# Default target
all: $(TARGET)

# Rule to create the output binary
$(TARGET): $(OBJS) $(METAL_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -fsanitize=address -g -o $@ $(OBJS) $(LDFLAGS)
	@echo "Built $(TARGET) with AddressSanitizer"

# Rule to compile Metal shaders into metallib
$(METAL_LIB): $(METAL_AIR_FILES)
	xcrun -sdk macosx metallib $^ -o $@
	@echo "Compiled Metal shaders into $@"

# Rule to compile each .metal shader into .air
$(BUILD_DIR)/%.air: %.metal
	@mkdir -p $(BUILD_DIR)
	xcrun -sdk macosx metal -O3 -c $< -o $@
	@echo "Compiled Metal shader $< -> $@"

# Rule to compile .cpp files to .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Phony targets
.PHONY: all clean