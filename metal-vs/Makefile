# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -I./metal-cpp -fobjc-arc
LDFLAGS = -framework Metal -framework MetalKit -framework AppKit -framework Foundation -framework QuartzCore

# Directories
SRC_DIR = .
CONTROL_DIR = ./control
VIEW_DIR = ./view
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/objs
CONTROL_OBJ_DIR = $(OBJ_DIR)/control
VIEW_OBJ_DIR = $(OBJ_DIR)/view

# Source files
SRCS = $(SRC_DIR)/main.cpp $(CONTROL_DIR)/app_delegate.cpp $(CONTROL_DIR)/view_delegate.cpp $(VIEW_DIR)/renderer.cpp
OBJS = $(OBJ_DIR)/main.o $(CONTROL_OBJ_DIR)/app_delegate.o $(CONTROL_OBJ_DIR)/view_delegate.o $(VIEW_OBJ_DIR)/renderer.o

# Output binary
TARGET = $(BUILD_DIR)/metal-vs

# Default target
all: $(TARGET)

# Rule to create the output binary
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built $(TARGET)"

# Rule to compile .cpp files to .o files
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(CONTROL_OBJ_DIR)/%.o: $(CONTROL_DIR)/%.cpp
	@mkdir -p $(CONTROL_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(VIEW_OBJ_DIR)/%.o: $(VIEW_DIR)/%.cpp
	@mkdir -p $(VIEW_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

# Phony targets
.PHONY: all clean