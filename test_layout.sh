nvcc gemv_fp16.cu -std=c++17 -arch=sm_80 -I /home/tyt/cutlass2/include/ -o fp16_layout.out

./fp16.out 1 16384 16384
./fp16_layout.out 1 16384 16384
./fp16.out 1 16384 16384
./fp16_layout.out 1 16384 16384

./fp16.out 2 16384 16384
./fp16_layout.out 2 16384 16384
./fp16.out 2 16384 16384
./fp16_layout.out 2 16384 16384

./fp16.out 4 16384 16384
./fp16_layout.out 4 16384 16384
./fp16.out 4 16384 16384
./fp16_layout.out 4 16384 16384

./fp16.out 8 16384 16384
./fp16_layout.out 8 16384 16384
./fp16.out 8 16384 16384
./fp16_layout.out 8 16384 16384

