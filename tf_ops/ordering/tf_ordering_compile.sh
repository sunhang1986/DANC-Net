#/bin/bash
/usr/local/cuda-8.0/bin/nvcc tf_ordering_g.cu -o tf_ordering_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_ordering.cpp tf_ordering_g.cu.o -o tf_ordering_so.so -shared -fPIC -I /../anaconda3/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0