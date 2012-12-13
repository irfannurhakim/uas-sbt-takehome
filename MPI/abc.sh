mpic++ $1.cpp -o $1
scp $1 167.205.32.202:~/$1
#scp $1 167.205.32.203:~/$1
scp $1 167.205.32.204:~/$1
scp $1 167.205.32.205:~/$1
scp $1 167.205.32.201:~/$1
time mpirun -np 10 --hostfile .mpi_hostfile $1 
