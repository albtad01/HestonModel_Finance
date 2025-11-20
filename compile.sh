nvcc -o heston_1 heston_1.cu -lcurand -arch=sm_70
./heston_1

nvcc -o heston_2 heston_2.cu -lcurand -arch=sm_70
./heston_2

nvcc -o heston_3 heston_3.cu -lcurand -arch=sm_70
./heston_3