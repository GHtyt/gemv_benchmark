nvcc gemv_w4a16.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o w4a16.out
nvcc gemv_w8a16.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o w8a16.out
nvcc gemv_w4a16g.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o w4a16g.out
nvcc gemv_fp16.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o fp16.out
nvcc gemv_int8.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o int8.out