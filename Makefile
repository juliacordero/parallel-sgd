all: serial_bgd parallel_bgd

serial_bgd: serial_bgd.c
	gcc -fopenmp serial_bgd.c -o serial_bgd

parallel_bgd: parallel_bgd.c
	gcc -fopenmp parallel_bgd.c -o parallel_bgd

clean:
	rm -f serial_bgd parallel_bgd