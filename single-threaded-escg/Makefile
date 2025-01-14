# Compiler and flags
CXX = clang++                          
CXXFLAGS = -std=c++17 -Wall -Wno-deprecated-declarations 

# Paths for includes
# - Include path for matplotlib-cpp
# - Include path for Python 3.11 headers
# - Include path for NumPy headers (arrayobject.h)
INCLUDE_FLAGS = -Isingle-threaded-escg/external/matplotlib-cpp \
                -I/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/include/python3.11 

# Library linking flags
# - Path to Python 3.11 libraries
# - Link Python 3.11 shared library, CoreFoundation framework, and dynamic loader library (ldl)
LDFLAGS = -L/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib \
          -lpython3.11 -ldl -framework CoreFoundation

# Target and sources
TARGET = escg                        
SRCS = single-threaded-escg.cpp simulation.cpp visualise.cpp  

# Default rule to build the target
all: $(TARGET)                       

# Rule to build the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Clean rule to remove the built executable
clean:
	rm -f $(TARGET)                     


# Note:
# - To find the NumPy include path, you can use the following command:
#   python3 -c "import numpy; print(numpy.get_include())"

# pip3 install numpy matplotlib


# pip install --upgrade pip && pip3 install numpy matplotlib
