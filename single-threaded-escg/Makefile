# Compiler and compiler flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Isingle-threaded-escg/external/matplotlib-cpp $(shell python3-config --includes)
# Target executable name
TARGET = escg

# Source files
SRCS = single-threaded-escg.cpp simulation.cpp visualise.cpp

# Default rule to build the target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(shell python3-config --ldflags)

# Clean rule to remove the built executable
clean:
	rm -f $(TARGET)