#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


// Function to call your MPI program and generate output matrix
void generate_output_matrix(double input1, double input2, double input3, double input4, FILE *file) {
    // Call your MPI program here to generate the output matrix
    char command[200];
    sprintf(command, "mpiexec -n 2 ./a.out %.2f %.2f %.2f %.2f", input1, input2, input3, input4);
    FILE *output_pipe = popen(command, "r");
    if (!output_pipe) {
        printf("Error: Failed to execute command\n");
        return;
    }

    // Read and write output matrix to CSV file
    char buffer[1000];
    while (fgets(buffer, sizeof(buffer), output_pipe) != NULL) {
        fputs(buffer, file);
    }

    pclose(output_pipe);
}

int main() {
    // Open CSV file to store data
    FILE *csv_file = fopen("data.csv", "w");
    if (!csv_file) {
        printf("Error: Could not open file data.csv\n");
        return -1;
    }


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int i1, i2, i3, i4;

    // Loop through each input value combination from 1 to 5
    for (int i1 = 0; i1 <= 3; i1++) {
        for (int i2 = 0; i2 <= 3; i2++) {
            for (int i3 = 0; i3 <= 3; i3++) {
                for (int i4 = 0; i4 <= 4; i4++) {
                    double input1 = (1.0*i1)/10.0;
                    double input2 = (1.0*i2)/10.0;
                    double input3 = (1.0*i3)/10.0;
                    double input4 = (1.0*i4)/10.0;

                    // Call function to generate output matrix and write it to CSV file
                    generate_output_matrix(input1, input2, input3, input4, csv_file);
                }
            }
        }
    }

    // Close CSV file
    fclose(csv_file);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %f seconds\n", elapsed);
    printf("Data stored in data.csv\n");

    return 0;
}
