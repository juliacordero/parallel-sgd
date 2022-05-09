# parallel-sgd

How to compile our code: 

serial_bgd:
gcc -fopenmp serial_bgd.c -o serial_bgd

parallel_bgd:
gcc -fopenmp parallel_bgd.c -o parallel_bgd

parallel_sgd_1_epoch:
gcc -fopenmp parallel_sgd_1_epoch.c -o parallel_sgd_1_epoch

parallel_sgd_S_epoch:
gcc -fopenmp parallel_sgd_S_epoch.c -o parallel_sgd_S_epoch

How to run our code:
./serial_bgd
./parallel_bgd [number of threads goes here as CL arg=1, 2, 4, 8, 16]
./parallel_sgd_1_epoch [number of threads goes here as CL arg=1, 2, 4, 8, 16]
./parallel_sgd_S_epoch [number of threads goes here as CL arg=1, 2, 4, 8, 16]
