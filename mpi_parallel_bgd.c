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
    // TODO(julia): better load balancing that just integer div
    // TODO(julia): also need to account for remainder in the div
    int num_samples_per_process = num_samples / size;
    int x_part[num_samples_per_process], y_part[num_samples_per_process];
    // TODO(julia): I don't know if this works
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

	    // Measure start of TOTAL time
	    if( clock_gettime(CLOCK_REALTIME, &total_start) == -1) { perror("clock gettime"); }
	    // Measure start of COMMUNICATION time
	    if( clock_gettime(CLOCK_REALTIME, &comm_start) == -1) { perror("clock gettime"); }
    }
    // TODO(julia): you moved a ton of stuff into process rank == 0, so you'll need to broadcast the necessary vars
    // to all processes

    // ********START EXECUTION OF GRADIENT DESCENT********
    // TODO(julia): Investigate MPI_Scatterv()
    MPI_Scatter(x, num_samples_per_process, MPI_DOUBLE, &x_part, num_samples_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, num_samples_per_process, MPI_DOUBLE, &y_part, num_samples_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
		// Measure end of COMMUNICATION time
	    if( clock_gettime( CLOCK_REALTIME, &comm_stop) == -1 ) { perror("clock gettime");}		
	    comm_time += (comm_stop.tv_sec - comm_start.tv_sec)+ (double)(comm_stop.tv_nsec - comm_start.tv_nsec)/1e9;
    }

    // COMPUTE PREVIOUS MEAN SQUARE ERROR
    // each process will compute its portion of prev_MSE in parallel
    double partial_prev_MSE[1];
    partial_prev_MSE[0] = 0;
    for (i = 0; i < num_samples_per_process; i++) {
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
    	for (i = 0; i < num_samples_per_process; i++) {
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
        for (i = 0; i < num_samples_per_process; i++) {
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