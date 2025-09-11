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

# # NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
# default:
# 	c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so