#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<math.h>
#include <stdbool.h>
#define PI 3.141593


double error(int N, double phit_1[][N], double phit[][N])
  {
    double norm = 0.0;
    for(int j = 0; j < N; j++){
      for(int i = 0; i < N; i++){
        norm += pow((phit[i][j] - phit_1[i][j]),2);
    }
  }
  return sqrt(norm);
  }

int main(int argc, char* argv[])
{
  double delta = 0.1;
  int N = (int) 2.0/delta + 1;
  double T1, T2, T3, T4;

   if (argc == 5)
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
  
  double phit[N][N];
  double phit_1[N][N];
  
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){

        if(i == 0){
            phit[i][j] = T1;
            phit_1[i][j] = T1;
            }

        else if(j == N-1){
            phit[i][j] = T2;
            phit_1[i][j] = T2;
            }

        else if(i == N-1){
            phit[i][j] = T3;
            phit_1[i][j] = T3;
            }

        else if(j == 0){
            phit[i][j] = T4;
            phit_1[i][j] = T4;
            }

        else{
        phit[i][j] = 0;
        phit_1[i][j] = 0;
        }
      }
    }
  
  
  int iteration = 0;

  clock_t start_time, end_time;
  double time_taken;
  bool condition = true;
  start_time = clock();

  while(condition){
  for(int i = 1; i < N-1; i++){
    for(int j = 1; j < N-1; j++){
      phit_1[i][j] = (phit[i+1][j] + phit[i-1][j] + phit[i][j+1] + phit[i][j-1])*0.25;
      }
    }
  condition = (error(N, phit_1, phit) > 0.0001);
  iteration ++;
  
  for(int j = 0; j < N; j++){
    for(int i = 0; i < N; i++){
	phit[i][j] = phit_1[i][j];
    }
  }
  
  }
  
  printf("%.2f, ",T1);
    printf("%.2f, ",T2);
    printf("%.2f, ",T3);
    printf("%.2f, ",T4);
    for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      printf("%.2f, ", phit[i][j]);
    }
  }

  printf("\n");
  
    return 0;
}