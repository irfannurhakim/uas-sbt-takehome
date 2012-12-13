/**********************************************************************************************
* Matrix Multiplication Program using MPI.
*
* Viraj Brian Wijesuriya - University of Colombo School of Computing, Sri Lanka.
* 
* Works with any type of two matrixes [A], [B] which could be multiplied to produce a matrix [c].
*
* Master process initializes the multiplication operands, distributes the muliplication 
* operation to worker processes and reduces the worker results to construct the final output.
*  
************************************************************************************************/

#include<stdio.h>
#include<mpi.h>
#define NUM_ROWS_A 12 //rows of input [A]
#define NUM_COLUMNS_A 12 //columns of input [A]
#define NUM_ROWS_B 12 //rows of input [B]
#define NUM_COLUMNS_B 12 //columns of input [B]
#define MASTER_TO_SLAVE_TAG 1 //tag for messages sent from master to slaves
#define SLAVE_TO_MASTER_TAG 4 //tag for messages sent from slaves to master
void makeAB(); //makes the [A] and [B] matrixes
void printArray(); //print the content of output matrix [C];

int rank; //process rank
int size; //number of processes
int i, j, k; //helper variables
double mat_a[NUM_ROWS_A][NUM_COLUMNS_A]; //declare input [A]
double mat_b[NUM_ROWS_B][NUM_COLUMNS_B]; //declare input [B]
double mat_result[NUM_ROWS_A][NUM_COLUMNS_B]; //declare output [C]
double start_time; //hold start time
double end_time; // hold end time
int low_bound; //low bound of the number of rows of [A] allocated to a slave
int upper_bound; //upper bound of the number of rows of [A] allocated to a slave
int portion; //portion of the number of rows of [A] allocated to a slave
MPI_Status status; // store status of a MPI_Recv
MPI_Request request; //capture request of a MPI_Isend

int N=1000;
float *x, *y, *z, *m;
float ax,dx;
float ay,dy;
float az,dz;
float invr, invr3, f;
double EPS = 3E4; 
float dt=10;
float *xnew, *ynew, *znew, *vx, *vy,*vz;

void make_matrix()
{
	x = (float*) malloc( N * sizeof(float));
    y = (float*) malloc( N * sizeof(float));
    z = (float*) malloc( N * sizeof(float));
    m = (float*) malloc( N * sizeof(float));
    xnew= (float*) malloc( N * sizeof(float));
    ynew= (float*) malloc( N * sizeof(float));
    znew= (float*) malloc( N * sizeof(float));
    vx = (float*) malloc( N * sizeof(float));
    vy = (float*) malloc( N * sizeof(float));
    vz = (float*) malloc( N * sizeof(float));
}
void init_matrix() {
    for (int i = 0; i < N; i++) {
		
        x[i]= (float) i+1;
        y[i]= (float) i+1;
        z[i]= (float) i+1;
        m[i]= (float) i+1;
		vx[i]= (float) 0;
		vy[i]= (float) 0;
		vz[i]= (float) 0;

    }
}
void print_matrix(float *m)
{
  int i, j;
  for (i = 0; i < N; i++) {
    printf("\n\t| ");
    
    printf("%.1f ", m[i]);
    printf("|");
  }
}
int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv); //initialize MPI operations
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes

    /* master initializes work*/
    if (rank == 0) {
        //makeAB();
		make_matrix()//deklarasi matrix
		init_matrix()//initialisasi matrix
        start_time = MPI_Wtime();
        for (i = 1; i < size; i++) {//for each slave other than the master
            portion = (N / (size - 1)); // calculate portion without master
            low_bound = (i - 1) * portion;
            if (((i + 1) == size) && ((N % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among slaves
                upper_bound = N; //last slave gets all the remaining rows
            } else {
                upper_bound = low_bound + portion; //rows of [A] are equally divisable among slaves
            }
            //send the low bound first without blocking, to the intended slave
            MPI_Isend(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &request);
            //next send the upper bound without blocking, to the intended slave
            MPI_Isend(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &request);
            //finally send the allocated row portion of [A] without blocking, to the intended slave
            MPI_Isend(&vx[low_bound], (upper_bound - low_bound), i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
			MPI_Isend(&vy[low_bound], (upper_bound - low_bound), i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
			MPI_Isend(&vz[low_bound], (upper_bound - low_bound), i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request);
        }
    }
    //broadcast [B] to all the slaves
    MPI_Bcast(&x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&z, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
    /* work done by slaves*/
    if (rank > 0) {
        //receive low bound from the master
        MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status);
        //next receive upper bound from the master
        MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &status);
        //finally receive row portion of [A] to be processed from the master
        MPI_Recv(&vx[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&vy[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&vz[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status);
        for (i = low_bound; i < upper_bound; i++) {//iterate through a given set of rows of [A]
            ax=0.0;
            ay=0.0;
            az=0.0;
            for (int j=0;j<N;j++)
            {
                dx=x[j]-x[i];
                dy=y[j]-y[i];
                dz=z[j]-z[i];
                invr = 1.0/sqrt(dx*dx + dy*dy + dz*dz+ EPS);
                invr3 = invr*invr*invr;
                f=m[j]*invr3;
                ax+=f*dx;
                ay+=f*dy;
                az+=f*dz;
            }
            xnew[i]=x[i]+ dt*vx[i] + 0.5*dt*dt*ax;
            ynew[i]=y[i]+ dt*vy[i] + 0.5*dt*dt*ay;
            znew[i]=z[i]+ dt*vz[i] + 0.5*dt*dt*az;
            vx[i]+=dt*ax;
            vy[i]+=dt*ay;
            vz[i]+=dt*az;
        }
        //send back the low bound first without blocking, to the master
        MPI_Isend(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &request);
        //send the upper bound next without blocking, to the master
        MPI_Isend(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &request);
        //finally send the processed portion of data without blocking, to the master
        MPI_Isend(&xnew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
		MPI_Isend(&ynew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
		MPI_Isend(&znew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request);
    }

    /* master gathers processed work*/
    if (rank == 0) {
        for (i = 1; i < size; i++) {// untill all slaves have handed back the processed data
            //receive low bound from a slave
            MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status);
            //receive upper bound from a slave
            MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &status);
            //receive processed data from a slave
            MPI_Recv(&xnew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&ynew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&znew[low_bound], (upper_bound - low_bound) , MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status);
        }
		for (int i=0; i<N; i++)
		{
        x[i]= xnew[i];
        y[i]= ynew[i];
        z[i]= znew[i];
		}
        end_time = MPI_Wtime();
        printf("\nRunning Time = %f\n\n", end_time - start_time);
        //printArray();
    }
    MPI_Finalize(); //finalize MPI operations
    return 0;
}

void makeAB()
{
    for (i = 0; i < NUM_ROWS_A; i++) {
        for (j = 0; j < NUM_COLUMNS_A; j++) {
            mat_a[i][j] = i + j;
        }
    }
    for (i = 0; i < NUM_ROWS_B; i++) {
        for (j = 0; j < NUM_COLUMNS_B; j++) {
            mat_b[i][j] = i*j;
        }
    }
}

void printArray()
{
    for (i = 0; i < NUM_ROWS_A; i++) {
        printf("\n");
        for (j = 0; j < NUM_COLUMNS_A; j++)
            printf("%8.2f  ", mat_a[i][j]);
    }
    printf("\n\n\n");
    for (i = 0; i < NUM_ROWS_B; i++) {
        printf("\n");
        for (j = 0; j < NUM_COLUMNS_B; j++)
            printf("%8.2f  ", mat_b[i][j]);
    }
    printf("\n\n\n");
    for (i = 0; i < NUM_ROWS_A; i++) {
        printf("\n");
        for (j = 0; j < NUM_COLUMNS_B; j++)
            printf("%8.2f  ", mat_result[i][j]);
    }
    printf("\n\n");
}