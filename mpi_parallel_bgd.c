#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define num_samples 96453

int main(int argc, char** argv) {
	printf("hi\n");
    int size, rank, i;
    // Read in the number of processes from .sl file
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    // Our two data arrays that will get split up amongst the processes using MPI_Scatter
    double x[num_samples], y[num_samples];
    
    // Gives us the minimum number of samples that each process will perform computation on.
    // For example, if there are 4 processes and 7 samples, each process will at minimum
    // do BGD on 7 / 4 = 1 sample.
    int counts[size] = {num_samples / size};
    int displacements[size];
    int max_data_sent_to_process = 0;

    double prev_MSE[1], gradb[1], gradm[1], curr_MSE[1];
    bool converged[1] = {false};
    int iter = 0;
	// initial weights
	double m[1], b[1] = {0};
	// Set up for gradient descent
	double alpha = 0.01;
	double ep = 0.01;
	int max_iter = 1000000;
	double weights[2];
    // Set up timer for TOTAL EXECUTION TIME of gradient descent
	struct timespec total_start, total_stop; 
	double total_time;
	// Set up timer for TOTAL COMMUNICATION TIME of gradient descent
	struct timespec comm_start, comm_stop; 
	double comm_time = 0;

    if (rank == 0) {
    	// read in x and y data
	    FILE *xdata;
	    FILE *ydata;
	    if (!(xdata=fopen("xdata.txt", "r"))) {
	        printf("cannot open file xdata\n");
	        return 1;
	    }
	    if (!(ydata=fopen("ydata.txt", "r"))) {
	        printf("cannot open file ydata\n");
	        return 1;
	    }
	    for (int i = 0; i < num_samples; i++) {
	        char xstr[64];
	        char ystr[64];

	        fgets(xstr, 64, xdata);
	        fgets(ystr, 64, ydata);
	        x[i] = atof(xstr);
	        y[i] = atof(ystr);
	    }
	    // Fully update counts array, which tells us how to divide the samples up by each process in a more
	    // load-balanced way.
	    int k;
	    int remainder = num_samples % size;
	    if (size > 1) {
	    	for (k = 0; k < remainder; k++) { 
		    	counts[k]++;
		    	if (counts[k] > max_data_sent_to_process) { max_data_sent_to_process = counts[k]; }
		    }
	    } else {
	    	max_data_sent_to_process = num_samples;
	    }
	    // Fully update the displacements array
	    displacements[0] = 0;
	    if (size > 1) {
	    	for (k = 1; k < size; k++) {
	    		// The spot where each process starts reading from should be where the previous process 
	    		// leaves off from. For example, process 0 starts reading at index 0, process 1 stars where
	    		// process 0 leaves off (at "n/p").
	    		displacements[k] = counts[k-1];
	    	}
	    }

	    // Measure start of TOTAL time
	    if( clock_gettime(CLOCK_REALTIME, &total_start) == -1) { perror("clock gettime"); }
	    // Measure start of COMMUNICATION time
	    if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
    }

    // ********START EXECUTION OF GRADIENT DESCENT********
    double x_part[max_data_sent_to_process], y_part[max_data_sent_to_process];
    MPI_Scatterv(x, counts, displacements, MPI_DOUBLE, &x_part, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, counts, displacements, MPI_DOUBLE, &y_part, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
		// Measure end of COMMUNICATION time
	    if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
	    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
    }

    // COMPUTE PREVIOUS MEAN SQUARE ERROR
    // each process will compute its portion of prev_MSE in parallel
    double partial_prev_MSE[1];
    partial_prev_MSE[0] = 0;
    // counts[rank] tells us how much data was sent to *this* particular process
    for (i = 0; i < counts[rank]; i++) {
        partial_prev_MSE[0] += (m[0]*x_part[i] + b[0] - y_part[i])*(m[0]*x_part[i] + b[0] - y_part[i]);
    }

    if (rank == 0) {
    	// Measure start of COMMUNICATION time
    	if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
    }
    
    // P0 uses MPI_SUM reduction to sum all partial sums
	MPI_Reduce(partial_prev_MSE, prev_MSE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		// Measure end of COMMUNICATION time
	    if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
	    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
	}
    
    while (!converged[0]) {
    	// COMPUTE GRADIENT
    	// every process computes on their own set of data
    	double partial_gradb[1], partial_gradm[1];
    	partial_gradm[0] = 0;
    	partial_gradb[0] = 0;
    	for (i = 0; i < counts[rank]; i++) {
            partial_gradb[0] += (b[0] + m[0]*x_part[i] - y_part[i]);
            partial_gradm[0] += (b[0] + m[0]*x_part[i] - y_part[i])*x_part[i];
        }

        if (rank == 0) {
	        // Measure start of COMMUNICATION time
	    	if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
	    }
    	// P0 uses MPI_SUM reduction to sum all partial sums
		MPI_Reduce(partial_gradb, gradb, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(partial_gradm, gradm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if (rank == 0) {
			// Measure end of COMMUNICATION time
		    if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
		    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;

		    // update weights globally
	    	b[0] = b[0] - alpha*gradb[0];
	        m[0] = m[0] - alpha*gradm[0];

	        if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
	        MPI_Bcast(b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	        MPI_Bcast(m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	        if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
		    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
		}

        // COMPUTE CURRENT MEAN SQUARE ERROR
        double partial_curr_MSE[1] = {0};
        for (i = 0; i < counts[rank]; i++) {
            partial_curr_MSE[0] += (m[0]*x_part[i] + b[0] - y_part[i])*(m[0]*x_part[i] + b[0] - y_part[i]);
        }

        if (rank == 0) {
	        // Measure start of COMMUNICATION time
	    	if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
    	}
    	// P0 uses MPI_SUM reduction to sum all partial sums
		MPI_Reduce(partial_curr_MSE, curr_MSE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0) {
			// Measure end of COMMUNICATION time
		    if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
		    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
    	
		    // CONVERGENCE CALCULATIONS DONE IN PROCESS 0
		    if (fabs(curr_MSE[0] - prev_MSE[0]) <= ep) {
	            printf("Converged\n");
	            printf("Num iterations: %d\n", iter);
	            converged[0] = true;
	        }
	        prev_MSE[0] = curr_MSE[0];
	        iter++;

	        if (iter == max_iter) {
	            printf("Max iterations exceeded\n");
	            converged[0] = true;
	        }

	        // Let all other processes know what's happening
	        // Measure start of COMMUNICATION time
	    	if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
	        MPI_Bcast(converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	        if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
		    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
    	}
    }

    weights[0] = b[0];
    weights[1] = m[0];

    if (rank == 0) {
    	// Measure end time
	    if( clock_gettime( CLOCK_REALTIME, &total_stop) == -1 ) { perror("clock gettime");}		
	    total_time = (total_stop.tv_sec - total_start.tv_sec)+ (double)(total_stop.tv_nsec - total_start.tv_nsec)/1e9;

	    printf("b = %f, m = %f\n Execution time = %f ms\n Communication time = %f ms\n", weights[0], weights[1], total_time*1000, comm_time*1000);
    }
    

    MPI_Finalize();
    return 0;
}