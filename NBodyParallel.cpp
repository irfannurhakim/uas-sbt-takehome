#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <omp.h>
using namespace std;

int N=1600;
float *x, *y, *z, *m;

void init_matrix(float *a, float *b, float *c, float *m) {
    for (int i = 0; i < N; i++) {
        a[i]= (float) i+1;
        b[i]= (float) i+1;
        c[i]= (float) i+1;
        m[i]= (float) i+1;
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
int main()
{
    int j,i;
    clock_t before, after;
    float time_serial;
    float ax,dx;
    float ay,dy;
    float az,dz;
    float invr, invr3, f;
    double EPS = 3E4; 
    float dt=0.1;
    float *xnew, *ynew, *znew, *vx, *vy,*vz;
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
    
    init_matrix(x,y,z,m);
    omp_set_dynamic(false);
    omp_set_num_threads(6);
    //before = clock();
    double start = omp_get_wtime( );
	for(int k=0;k<1000;k++){
            # pragma omp parallel for shared (xnew,znew,ynew,vx,vy,vz,x,y,z) private (f,ax,ay,az,dx,dy,dz,invr,invr3)
	    for(int i=0;i<N;i++)
	    {
	            ax=0.0;
	            ay=0.0;
	            az=0.0;
	            for ( j=0;j<N;j++)
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
    
	    for (int i=0; i<N; i++)
	    {
	        x[i]= xnew[i];
	        y[i]= ynew[i];
	        z[i]= znew[i];
	    }
	}
	
	
    //after = clock();
    //print_matrix(x);
    double end = omp_get_wtime( );
    printf("OpenMP Matrx  %.16g  seconds\n",end - start);
    //time_serial= (float)(after - before)/ CLOCKS_PER_SEC;
   //printf(" serial mxm = %.3f seconds\n", time_serial );
    
    //system("pause");
}
