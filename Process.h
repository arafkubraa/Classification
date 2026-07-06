#pragma once
#include <cmath>
#include <cstdlib>


float* Add_Data(float* sample, int Size, float* x, int Dim);
float* Add_Labels(float* Labels, int Size, int label);
float* init_array_random(int len);


void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std);
int Predict_Binary_Class(const float* test_sample, const float* weights, float bias, int inputDim);
int Test_Forward_Multiclass(const float* test_sample, const float* weights, const float* bias, int class_count, int inputDim);


int Predict_MLP_Multiclass(const float* test_sample,
                           int* layer_sizes, int num_layers,
                           float** weights, float** biases);


void Train_Perceptron_Binary_C(float* samples, float* targets,
                               float* weights, float* bias,
                               int numSample, int inputDim,
                               float* mean_out, float* std_out,
                               float* normalized_samples_out,
                               float* epoch_errors_out,
                               int* epochs_run_out);

void Train_Perceptron_Multiclass_C(float* samples, float* targets,
                                   float* weights, float* bias,
                                   int numSample, int inputDim, int class_count,
                                   float* mean_out, float* std_out,
                                   float* normalized_samples_out,
                                   float* epoch_errors_out, int* epochs_run_out);


void Train_MLP_Multiclass_C(float* samples, float* targets,
                            int numSample,
                            int* layer_sizes, int num_layers,
                            float** weights, float** biases,
                            float* mean_out, float* std_out,
                            float* normalized_samples_out,
                            float* epoch_errors_out, int* epochs_run_out);