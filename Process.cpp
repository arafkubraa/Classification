#include "Process.h"
#include <cstdlib> 

float* Add_Data(float* sample, int Size, float* x, int Dim) {
    float* temp;
    temp = new float[Size * Dim];
    for (int i = 0; i < (Size - 1) * Dim; i++)
        temp[i] = sample[i];
    for (int i = 0; i < Dim; i++)
        temp[(Size - 1) * Dim + i] = x[i];
    //delete[] sample; 
    return temp;
}

float* Add_Labels(float* Labels, int Size, int label) {
    float* temp;
    temp = new float[Size];
    for (int i = 0; i < Size - 1; i++)
        temp[i] = Labels[i];
    temp[Size - 1] = float(label);
    //delete[] Labels; 
    return temp;
}

// üçü de yeni dizi oluşturur ve geri döndürür.
float* init_array_random(int len) {
    float* arr = new float[len];
    for (int i = 0; i < len; i++)
        arr[i] = ((float)rand() / RAND_MAX) - 0.5f;
    return arr;
}