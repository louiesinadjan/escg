# Compiler and compilation flags
CXX       = clang++
CXXFLAGS  = -std=c++17 -Wall -Wno-deprecated-declarations -pg

# Attempt to use --embed; if unavailable, fallback to normal --ldflags
EMBED_LDFLAGS = $(shell (python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags))

# Dynamically fetch other needed paths
PYTHON_INCLUDE  := $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
NUMPY_INCLUDE   := $(shell python3 -c "import numpy; print(numpy.get_include())")

# Additional include paths:
INCLUDE_FLAGS = -Isingle-threaded-escg/external/matplotlib-cpp \
                -I$(PYTHON_INCLUDE) \
                -I$(NUMPY_INCLUDE)

# Some systems require -undefined dynamic_lookup to load Python symbols at runtime on macOS
# We combine that with the embed or fallback flags.
LDFLAGS = $(EMBED_LDFLAGS) -undefined dynamic_lookup

TARGET = escg
SRCS   = single-threaded-escg.cpp simulation.cpp visualise.cpp timer.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS) -pg

clean:
	rm -f $(TARGET)

# -------------------------------------------------------------------------
# USAGE:
# 1) Ensure "python3" in your shell is the pyenv Python 3.11.11 (arm64).
#    which python3
#    python3 --version
#    file "$(pyenv which python3)"    # Should say Mach-O 64-bit arm64
#
# 2) pip-install NumPy & Matplotlib in that environment:
#    python3 -m pip install --upgrade pip
#    python3 -m pip install numpy matplotlib
#
# 3) make clean && make
#    ./escg
#
# If you still see missing symbols:
#    - Double-check that python3-config --ldflags --embed returns valid arm64 flags.
#    - Make sure the Python library is truly arm64.
# -------------------------------------------------------------------------