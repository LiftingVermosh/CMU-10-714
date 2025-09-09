PYTHON_INCLUDE := $(shell python -m pybind11 --includes)
PYTHON_LIB_DIR := E:/Miniconda3/envs/CMU-10-714/libs
PYTHON_LIB := -lpython39

default:
	c++ -O3 -Wall -shared -std=c++11 -fPIC \
		$(PYTHON_INCLUDE) \
		src/simple_ml_ext.cpp \
		-o src/simple_ml_ext.pyd \
		-L$(PYTHON_LIB_DIR) $(PYTHON_LIB) \
		-static-libgcc -static-libstdc++
