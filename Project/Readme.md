# Parallelization of DNNs for Steady-State Heat Conduction Problems

This project conducts a comparative study of neural network training time with and without parallelization using OpenACC and MPI. The stages are as follows:

- Generating a steady-state heat conduction problem
- Calculating the steady-state temperatures, with parallelization using MPI
- Training the neural network on the generated data points, with parallelization using OpenACC
- Evaluating the time taken and speed-up with varying data points, epochs, and number of processors.
