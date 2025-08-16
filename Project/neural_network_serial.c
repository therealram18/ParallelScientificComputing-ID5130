#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 4
#define OUTPUT_SIZE 441
#define HIDDEN_SIZE 20
#define LEARNING_RATE 1
#define EPOCHS 200

// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Mean squared error loss function
double mean_squared_error(double output[OUTPUT_SIZE], double target[OUTPUT_SIZE]) {
    double error = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        error += pow(output[i] - target[i], 2);
    }
    return error / OUTPUT_SIZE;
}

// Forward propagation
void forward_propagation(double inputs[INPUT_SIZE], double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_o[OUTPUT_SIZE]) {
    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += inputs[j] * weights_ih[j][i];
        }
        hidden[i] += bias_h[i];
        hidden[i] = sigmoid(hidden[i]);
    }
    
    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * weights_ho[j][i];
        }
        output[i] += bias_o[i];
        output[i] = sigmoid(output[i]);
    }
}

// Backpropagation
void backpropagation(double inputs[INPUT_SIZE], double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE], double target[OUTPUT_SIZE], double weights_ih[INPUT_SIZE][HIDDEN_SIZE], double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE], double bias_h[HIDDEN_SIZE], double bias_o[OUTPUT_SIZE]) {
    // Calculate output layer errors
    double output_errors[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_errors[i] = target[i] - output[i];
    }

    // Calculate output layer gradients
    double output_gradients[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_gradients[i] = output_errors[i] * sigmoid_derivative(output[i]);
    }

    // Update output layer weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] += LEARNING_RATE * output_gradients[j] * hidden[i];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] += LEARNING_RATE * output_gradients[i];
    }

    // Calculate hidden layer errors
    double hidden_errors[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_errors[i] += output_gradients[j] * weights_ho[i][j];
        }
    }

    // Calculate hidden layer gradients
    double hidden_gradients[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_gradients[i] = hidden_errors[i] * sigmoid_derivative(hidden[i]);
    }

    // Update hidden layer weights and biases
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] += LEARNING_RATE * hidden_gradients[j] * inputs[i];
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias_h[i] += LEARNING_RATE * hidden_gradients[i];
    }
}


// Load data from CSV file
void load_data(const char *filename, double inputs[][INPUT_SIZE], double outputs[][OUTPUT_SIZE], int num_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < num_samples; i++) {
        // Read input values
        for (int j = 0; j < INPUT_SIZE; j++) {
            if (fscanf(file, "%lf,", &inputs[i][j]) != 1) {
                printf("Error reading input values from file %s\n", filename);
                exit(1);
            }
        }

        // Read target output values
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (fscanf(file, "%lf,", &outputs[i][j]) != 1) {
                printf("Error reading output values from file %s\n", filename);
                exit(1);
            }
        }
    }

    fclose(file);
}


int main() {
    double inputs[1296][INPUT_SIZE];
    double outputs[1296][OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    int num_samples = 1296;

    // Read input-output pairs from CSV file
    load_data("data.csv", inputs, outputs, num_samples);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Initialize weights and biases randomly
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias_h[HIDDEN_SIZE];
    double bias_o[OUTPUT_SIZE];
    

    // Initialize weights and biases randomly
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_ih[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        } 
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            weights_ho[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        bias_h[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_o[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
        
    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < num_samples; i++) {

           
            forward_propagation(inputs[i], hidden, output, weights_ih, weights_ho, bias_h, bias_o);
            backpropagation(inputs[i], hidden, output, outputs[i], weights_ih, weights_ho, bias_h, bias_o);
            

            // Calculate loss for this iteration
            double loss = mean_squared_error(output, outputs[i]);
            total_loss += loss;
        }
        double avg_loss = total_loss / num_samples; // Calculate average loss per sample
        if(epoch % 10 == 0)
        printf("Epoch %d - Average Loss: %f, ", epoch + 1, avg_loss);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    

    // Print some inference results
    int idx[5] = {247, 342, 827, 560, 959};
    

    double inference_output[OUTPUT_SIZE];
    for(int m = 0; m < 5; m++){

    forward_propagation(inputs[idx[m]], hidden, inference_output, weights_ih, weights_ho, bias_h, bias_o);
    printf("\nInference Output:\n");

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%.2f %.2f\n", inference_output[i], outputs[idx[m]][i]);
        }
    printf("\n");

    }

    printf("Elapsed time: %f seconds\n", elapsed);

    return 0;
}
