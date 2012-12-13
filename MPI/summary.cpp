#include<iostream>
#include<mpi.h>
#include<ctime>
using namespace std;



/*void sleep(unsigned int mseconds){
	clock_t goal = mseconds + clock();
	while(goal > clock());
}*/



int main(int argc, char ** argv){
    	//clock_t before, after;
    	//float time_serial;
	double before, after, time, taccum;	
int mynode, totalnodes;
	int sum,startval,endval,accum;// taccum;
	int numtasks, rank;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	int test;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	sum = 0; // zero sum for accumulation
	time = 0;
	startval = 100*mynode/totalnodes+1;
	endval = 100*(mynode+1)/totalnodes;
        MPI_Get_processor_name(processor_name, &name_len);
	before = MPI_Wtime();
	cout<<sum<<" "<<time<<" "<<endl;
	for(int i=startval;i<=endval;i=i+1){
		sum = sum + i;
	}
	after = MPI_Wtime();		
	time = after - before;
	time = time  + (after - before);
	cout<<sum<<" "<<time<<" "<<endl;	
	//cout<<"Proc Name : "<<processor_name<<" startval : "<<startval<<" endval :"<<endval<<" my node : "<<mynode<<" time : "<<time<<endl;

        
	if(mynode!=0){
		MPI_Send(&sum,1,MPI_INT,0,1,MPI_COMM_WORLD);
		MPI_Send(&time, 1, MPI_DOUBLE,0,1, MPI_COMM_WORLD);
	}
	else{
		for(int j=1;j<totalnodes;j=j+1){
			MPI_Recv(&accum,1,MPI_INT,j,1,MPI_COMM_WORLD,&status);
			MPI_Recv(&taccum,1,MPI_DOUBLE,j,1,MPI_COMM_WORLD,&status);
			time = time + taccum;
		
			sum = sum + accum;
		}
		

	}

	//after = clock();
	if(mynode == 0) {
		cout << "The sum from 1 to 1000 is: " << sum <<endl;
 			
	    printf(" paralel mxm = %lf seconds\n", time );
	}
	MPI_Finalize();
}
