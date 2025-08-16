#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<math.h>
#include <stdbool.h>
#include<mpi.h>
#define PI 3.141593

// double q(int i, int j, double delta, int ln, int my_id){
//   double x = -1.0 + (j*delta);
//   double y = -1.0 + ((my_id*ln + i)*delta);
//   return 2.0*(2-((x*x) + (y*y)));
// }

double error(int N, double phit_1[][N], double phit[][N], int ln)
  {
    double norm = 0.0;
    for(int i = 1; i < ln + 1; i++){
      for(int j = 0; j < N; j++){
        norm += pow((phit[i][j] - phit_1[i][j]),2);
    }
  }
  return norm;
  }


int main(int argc, char* argv[])
{

  int my_id, size, tag = 100;
  double L = 2.0;
  double delta = 0.1;
  double T1, T2, T3, T4;
  int iter = 0;
  
  if (argc == 5) //if k is required to be taken as input change argc = 3
    {
      T1 = atof(argv[1]);
      T2 = atof(argv[2]);
      T3 = atof(argv[3]);
      T4 = atof(argv[4]);
      
      
    }
  else
    {
      printf("\n A command line argument other than name of the executable is required...exiting the program..\n");
      return 1;
    }
  int N =  (int) L/delta + 1;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Barrier(MPI_COMM_WORLD);

  double start_time = MPI_Wtime();
  int ln = (int) (N / size);
  
  double phi[N][N];
  double local_phit[ln+2][N];
  double local_phit_1[ln+2][N];
  double local_error;
  double global_error;
  bool condition = true;


  for(int i = 0; i < ln + 2; i++){
    
    for(int j = 0; j < N; j++){
        if(i == ln+1 && my_id == size-1){
          if(j > 0 && j < N-1){
            local_phit[i][j] = T3;
            local_phit_1[i][j] = T3;
          }
          else if(j==0){
            local_phit[i][j] = T4;
            local_phit_1[i][j] = T4;
          }
          else{
            local_phit[i][j] = T2;
            local_phit_1[i][j] = T2;
          }
        }

        else{
            local_phit[i][j] = 0;
            local_phit_1[i][j] = 0;
        }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  while (condition){
    ////////////////send receive //////////////////////
  
  for (int i = 1; i < ln+1; i++){
    
        for (int j = 0; j < N ; j++){
          if(j > 0){
            if(j == N-1){
                local_phit_1[i][j] = T2;
            }

            else{
                local_phit_1[i][j] = (local_phit[i-1][j] + local_phit[i+1][j] + local_phit[i][j-1] + local_phit[i][j+1]  /*+ (pow(delta,2)*q(i, j, delta, ln, my_id))*/)*0.25;
            }

            if((my_id == 0 && i == 1 && j != N-1)){
                local_phit_1[i][j] = T1;
            }
          }
          else{
            local_phit_1[i][j] = T4;
          }
        }
      
   }

   if (my_id != size -1)
    {
      MPI_Send(&local_phit_1[ln][0], N, MPI_DOUBLE, my_id + 1, tag, MPI_COMM_WORLD);
    }
  if (my_id != 0){
    MPI_Recv(&local_phit_1[0][0], N, MPI_DOUBLE, my_id - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  if (my_id != 0){
    MPI_Send(&local_phit_1[1][0], N, MPI_DOUBLE, my_id - 1, tag, MPI_COMM_WORLD);
  }
  if(my_id != size - 1){
    MPI_Recv(&local_phit_1[ln+1][0], N, MPI_DOUBLE, my_id + 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

 
 local_error = error(N, local_phit_1, local_phit, ln);

 MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 global_error = sqrt(global_error);
 condition = (global_error > 0.0001);
 iter++;
 
 MPI_Barrier(MPI_COMM_WORLD);

 for(int i = 0; i < ln+2; i++){
    for(int j = 0; j < N; j++){
        local_phit[i][j] = local_phit_1[i][j];
       }
    }
}

MPI_Barrier(MPI_COMM_WORLD);

double end_time = MPI_Wtime();

/*if(my_id == 0){
    printf("number of iterations: %d \n", iter);
    printf("time taken %f in sec\n",end_time - start_time);
  }*/


  MPI_Gather(&local_phit_1[1][0], ln*N, MPI_DOUBLE, phi, ln*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  if(my_id == 0){
    for(int j = 0; j < N; j++){
      if(j==0){
        phi[N-1][j] = T4;
      }
      else if(j==N-1){
        phi[N-1][j] = T2;
      }
      else{
        phi[N-1][j] = T3;
      }
    }
  }
MPI_Barrier(MPI_COMM_WORLD);

  if(my_id==0){
    printf("%.2f, ",T1);
    printf("%.2f, ",T2);
    printf("%.2f, ",T3);
    printf("%.2f, ",T4);
    for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      printf("%.2f, ", phi[i][j]);
    }
  }
  printf("\n");
  }

  MPI_Finalize();

  return 0;
}
