#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

// size of local matrix
#define my_COLS  1000
#define my_ROWS  250

// define numerical error bound
#define MAX_TEMP_ERROR  0.01


double Temperature[my_ROWS+2][my_COLS+2];
double Temperature_last[my_ROWS+2][my_COLS+2];

void initialize(int my_PE_num); //INITIALIZE BOUNDARY CONDTION

int main(int argc, char **argv)
{

	int max_iteration = 10000000;
	int iteration = 1;
	double dt = 100;
	double my_dt = 100;

	int my_PE_num, vec2send, vec2receive;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

	initialize(my_PE_num);


	while(dt > MAX_TEMP_ERROR && iteration <= max_iteration){
		for (int i = 1; i < my_ROWS; ++i)
		{
			for (int j = 1; i < my_COLS; ++j)
			{
				Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +Temperature_last[i][j+1] + Temperature_last[i][j-1]);
			}
		}

		my_dt = 0.0;

		// update local max error
		for (int i = 1; i < my_ROWS; ++i)
		{
			for (int j = 1; i < my_COLS; ++j)
			{
				my_dt = fmax(fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
				Temperature_last[i][j] = Temperature[i][j];
			}
		}

		// barrier for local error update
		MPI_Barrier(MPI_COMM_WORLD);

		// update global error
		MPI_Allreduce(&my_dt, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


		// update local top
		if(my_PE_num < 3){
			MPI_Send(&Temperature[my_ROWS][0], my_COLS+2, MPI_DOUBLE, my_PE_num+1, 10, MPI_COMM_WORLD); // sending bottom padding row

		}

		if(my_PE_num > 0){
			MPI_Recv(&Temperature_last[0][0],my_COLS+2, MPI_DOUBLE, my_PE_num-1,10, MPI_COMM_WORLD, &status);
		}

		//update local bottom
		if (my_PE_num > 0){
			MPI_Send(&Temperature[1][0], my_COLS+2, MPI_DOUBLE, my_PE_num-1,5, MPI_COMM_WORLD);// send top padding row
		}

		if(my_PE_num < 3){
			MPI_Recv(&Temperature_last[my_ROWS+1][0], my_COLS+2, MPI_DOUBLE, my_PE_num+1, 5, MPI_COMM_WORLD, &status);
		}

		// barrier for padding row update;
		MPI_Barrier(MPI_COMM_WORLD);

	}




	MPI_Finalize();
	return 0;
}


void initialize(int my_PE_num){

	for (int i = 0; i <= my_ROWS+1; i++){
		for (int j = 0; j < my_COLS; j++){
			Temperature_last[i][j] = 0.0;
		}
	}


	// set right edge to linear increase;
	for (int i =0; i <= my_ROWS + 1; i++){
		Temperature_last[i][my_COLS+1] = (100.0/my_ROWS)*(i+(my_ROWS+1)*my_PE_num);
	}


	//set bottom edge to linear increase, bottom edge belongs to last PE
	if(my_PE_num == 4){
		for (int j = 0; j < my_COLS; ++j){
			Temperature_last[my_ROWS+1][j] = (100.0/my_COLS)*j;
		}
	}
}