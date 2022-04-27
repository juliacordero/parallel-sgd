#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define num_samples 96453

void gradient_descent(double alpha, double* x, double* y, double ep, int max_iter, double* weights)
{
    bool converged = false;
    int iter = 0;

    // initial weights
    double m = 0;
    double b = 0;

    // mean squared error
    double prev_MSE = 0;
    for (int i = 0; i < num_samples; i++)
    {
        prev_MSE += (m*x[i] + b - y[i])*(m*x[i] + b - y[i]);
    }

    while(!converged)
    {
        double gradb = 0;
        double gradm = 0;

        // compute gradient across all samples
        for (int i = 0; i < num_samples; i++)
        {
            gradb += (b + m*x[i] - y[i]);
            gradm += (b + m*x[i] - y[i])*x[i];
        }
        gradb = gradb/num_samples;
        gradm = gradm/num_samples;

        // update weights
        b = b - alpha*gradb;
        m = m - alpha*gradm;

        // compute mean squared error
        double curr_MSE = 0;
        for (int i = 0; i < num_samples; i++)
        {
            curr_MSE += (m*x[i] + b - y[i])*(m*x[i] + b - y[i]);
        }

        if (fabs(curr_MSE - prev_MSE) <= ep)
        {
            printf("Converged\n");
            printf("Num iterations: %d\n", iter);
            converged = true;
        }

        prev_MSE = curr_MSE;        // update error
        iter++;

        if (iter == max_iter)
        {
            printf("Max iterations exceeded\n");
            {
                converged = true;
            }
        }
    }

    weights[0] = b;
    weights[1] = m;
    
}

int main(int argc, char *argv[])
{
    double x[num_samples];
    double y[num_samples];

    // read in x and y data
    FILE *xdata;
    FILE *ydata;
    if (!(xdata=fopen("xdata.txt", "r")))
    {
        printf("cannot open file xdata\n");
        return 1;
    }
    if (!(ydata=fopen("ydata.txt", "r")))
    {
        printf("cannot open file ydata\n");
        return 1;
    }

    for (int i = 0; i < num_samples; i++)
    {
        char xstr[64];
        char ystr[64];

        fgets(xstr, 64, xdata);
        fgets(ystr, 64, ydata);
        x[i] = atof(xstr);
        y[i] = atof(ystr);
    }

    // run gradient descent
    double alpha = 0.001;
    double ep = 0.01;
    int max_iter = 1000000;
    double weights[2];

    struct timespec start, stop; 
    double time;

    // Measure start time
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }
    
    gradient_descent(alpha, x, y, ep, max_iter, weights);

    // Measure end time
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;


    printf("b = %f, m = %f\n Execution time = %f ms\n", weights[0], weights[1], time*1000);
}