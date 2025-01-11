# Compiler and compiler flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall

# Target executable name
TARGET = escg

# Source files
SRCS = single-threaded-escg.cpp

# Default rule to build the target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean rule to remove the built executable
clean:
	rm -f $(TARGET)