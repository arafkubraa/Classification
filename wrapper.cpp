#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Process.h"
#include <vector>

namespace py = pybind11;


py::capsule create_deleter(float* ptr) {
    return py::capsule(ptr, [](void *f) {
        delete[] reinterpret_cast<float*>(f);
    });
}


py::array_t<float> PyAdd_Data_Wrapper(const py::array_t<float>& samples_np, int current_size,
                                      const py::array_t<float>& new_sample_np, int inputDim)
{
    float* samples_ptr = (float*)samples_np.data();
    float* new_sample_ptr = (float*)new_sample_np.data();
    int new_size = current_size + 1;
    float* result_ptr = Add_Data(samples_ptr, new_size, new_sample_ptr, inputDim);
    return py::array_t<float>(new_size * inputDim, result_ptr, create_deleter(result_ptr));
}

py::array_t<float> PyAdd_Labels_Wrapper(const py::array_t<float>& targets_np, int current_size, int new_label)
{
    float* targets_ptr = (float*)targets_np.data();
    int new_size = current_size + 1;
    float* result_ptr = Add_Labels(targets_ptr, new_size, new_label);
    return py::array_t<float>(new_size, result_ptr, create_deleter(result_ptr));
}

py::array_t<float> PyInit_Array_Random_Wrapper(int len)
{
    float* result_ptr = init_array_random(len);
    return py::array_t<float>(len, result_ptr, create_deleter(result_ptr));
}


std::tuple<py::array_t<float>, py::array_t<float>>
PyCalculate_Z_Score_Wrapper(const py::array_t<float>& samples_np, int numSample, int inputDim)
{
    float* samples_ptr = (float*)samples_np.data();
    float* mean_ptr = new float[inputDim];
    float* std_ptr = new float[inputDim];
    Z_Score_Parameters(samples_ptr, numSample, inputDim, mean_ptr, std_ptr);
    return std::make_tuple(
        py::array_t<float>(inputDim, mean_ptr, create_deleter(mean_ptr)),
        py::array_t<float>(inputDim, std_ptr, create_deleter(std_ptr))
    );
}

int PyPredict_Binary_Class_Wrapper(const py::array_t<float>& test_sample_np,
                                   const py::array_t<float>& weights_np, float bias,
                                   int inputDim)
{
    const float* test_sample_ptr = (const float*)test_sample_np.data();
    const float* weights_ptr = (const float*)weights_np.data();
    return Predict_Binary_Class(test_sample_ptr, weights_ptr, bias, inputDim);
}

int PyPredict_Multiclass_Wrapper(const py::array_t<float>& test_sample_np,
                                 const py::array_t<float>& weights_np,
                                 const py::array_t<float>& bias_np,
                                 int inputDim, int class_count)
{
    const float* test_sample_ptr = (const float*)test_sample_np.data();
    const float* weights_ptr = (const float*)weights_np.data();
    const float* bias_ptr = (const float*)bias_np.data();
    return Test_Forward_Multiclass(test_sample_ptr, weights_ptr, bias_ptr, class_count, inputDim);
}


int PyPredict_MLP_Multiclass_Wrapper(const py::array_t<float>& test_sample_np,
                                     const std::vector<int>& layer_sizes,
                                     const std::vector<py::array_t<float>>& weights_np,
                                     const std::vector<py::array_t<float>>& biases_np)
{
    const float* test_sample_ptr = (const float*)test_sample_np.data();
    
    std::vector<float*> weights_ptr;
    for(const auto& arr : weights_np) {
        weights_ptr.push_back((float*)arr.data());
    }

    std::vector<float*> biases_ptr;
    for(const auto& arr : biases_np) {
        biases_ptr.push_back((float*)arr.data());
    }

    return Predict_MLP_Multiclass(test_sample_ptr, (int*)layer_sizes.data(), layer_sizes.size(), weights_ptr.data(), biases_ptr.data());
}



std::tuple<py::array_t<float>, float,
           py::array_t<float>, py::array_t<float>,
           py::array_t<float>, py::array_t<float>>
PyTrain_Perceptron_Binary_Wrapper(const py::array_t<float>& samples_np,
                                  const py::array_t<float>& targets_np,
                                  const py::array_t<float>& weights_np, float bias,
                                  int numSample, int inputDim)
{
    std::vector<float> Weights_vec(weights_np.data(), weights_np.data() + weights_np.size());
    float bias_copy = bias;
    float* mean_out = new float[inputDim];
    float* std_out = new float[inputDim];
    float* norm_samples_out = new float[numSample * inputDim];
    int max_iter = 10000;
    float* epoch_errors_out = new float[max_iter];
    int epochs_run = 0;

    Train_Perceptron_Binary_C(
        (float*)samples_np.data(), (float*)targets_np.data(),
        Weights_vec.data(), &bias_copy,
        numSample, inputDim,
        mean_out, std_out, norm_samples_out,
        epoch_errors_out, &epochs_run
    );

    py::array_t<float> new_weights_np = py::cast(Weights_vec);
    py::array_t<float> epoch_errors_np(epochs_run);
    std::memcpy(epoch_errors_np.mutable_data(), epoch_errors_out, epochs_run * sizeof(float));
    delete[] epoch_errors_out;

    return std::make_tuple(
        new_weights_np, bias_copy,
        py::array_t<float>(inputDim, mean_out, create_deleter(mean_out)),
        py::array_t<float>(inputDim, std_out, create_deleter(std_out)),
        py::array_t<float>(numSample * inputDim, norm_samples_out, create_deleter(norm_samples_out)),
        epoch_errors_np
    );
}

std::tuple<py::array_t<float>, py::array_t<float>,
           py::array_t<float>, py::array_t<float>,
           py::array_t<float>, py::array_t<float>>
PyTrain_Perceptron_Multiclass_Wrapper(const py::array_t<float>& samples_np,
                                      const py::array_t<float>& targets_np,
                                      const py::array_t<float>& weights_np,
                                      const py::array_t<float>& bias_np,
                                      int numSample, int inputDim, int class_count)
{
    std::vector<float> Weights_vec(weights_np.data(), weights_np.data() + weights_np.size());
    std::vector<float> Bias_vec(bias_np.data(), bias_np.data() + bias_np.size());

    float* mean_out = new float[inputDim];
    float* std_out = new float[inputDim];
    float* norm_samples_out = new float[numSample * inputDim];
    int max_iter = 10000;
    float* epoch_errors_out = new float[max_iter];
    int epochs_run = 0;

    Train_Perceptron_Multiclass_C(
        (float*)samples_np.data(), (float*)targets_np.data(),
        Weights_vec.data(), Bias_vec.data(),
        numSample, inputDim, class_count,
        mean_out, std_out, norm_samples_out,
        epoch_errors_out, &epochs_run
    );

    py::array_t<float> new_weights_np = py::cast(Weights_vec);
    py::array_t<float> new_bias_np = py::cast(Bias_vec);
    py::array_t<float> epoch_errors_np(epochs_run);
    std::memcpy(epoch_errors_np.mutable_data(), epoch_errors_out, epochs_run * sizeof(float));
    delete[] epoch_errors_out;

    return std::make_tuple(
        new_weights_np, new_bias_np,
        py::array_t<float>(inputDim, mean_out, create_deleter(mean_out)),
        py::array_t<float>(inputDim, std_out, create_deleter(std_out)),
        py::array_t<float>(numSample * inputDim, norm_samples_out, create_deleter(norm_samples_out)),
        epoch_errors_np
    );
}

// MLP Multiclass Training Wrapper for flexible layers
std::tuple<std::vector<py::array_t<float>>, std::vector<py::array_t<float>>, 
           py::array_t<float>, py::array_t<float>, 
           py::array_t<float>, py::array_t<float>>
PyTrain_MLP_Multiclass_Wrapper(const py::array_t<float>& samples_np,
                               const py::array_t<float>& targets_np,
                               const std::vector<int>& layer_sizes,
                               const std::vector<py::array_t<float>>& weights_np,
                               const std::vector<py::array_t<float>>& biases_np,
                               int numSample)
{
    int inputDim = layer_sizes[0];
    
    std::vector<std::vector<float>> weights_vecs;
    std::vector<float*> weights_ptrs;
    for(const auto& arr : weights_np) {
        weights_vecs.emplace_back(arr.data(), arr.data() + arr.size());
        weights_ptrs.push_back(weights_vecs.back().data());
    }

    std::vector<std::vector<float>> biases_vecs;
    std::vector<float*> biases_ptrs;
    for(const auto& arr : biases_np) {
        biases_vecs.emplace_back(arr.data(), arr.data() + arr.size());
        biases_ptrs.push_back(biases_vecs.back().data());
    }
    
    float* mean_out = new float[inputDim];
    float* std_out = new float[inputDim];
    float* norm_samples_out = new float[numSample * inputDim];
    int max_iter = 10000;
    float* epoch_errors_out = new float[max_iter];
    int epochs_run = 0;

    Train_MLP_Multiclass_C(
        (float*)samples_np.data(), (float*)targets_np.data(),
        numSample, (int*)layer_sizes.data(), layer_sizes.size(),
        weights_ptrs.data(), biases_ptrs.data(),
        mean_out, std_out, norm_samples_out,
        epoch_errors_out, &epochs_run
    );

    std::vector<py::array_t<float>> new_weights_np;
    for(const auto& vec : weights_vecs) {
        new_weights_np.push_back(py::cast(vec));
    }

    std::vector<py::array_t<float>> new_biases_np;
    for(const auto& vec : biases_vecs) {
        new_biases_np.push_back(py::cast(vec));
    }

    py::array_t<float> epoch_errors_np(epochs_run);
    std::memcpy(epoch_errors_np.mutable_data(), epoch_errors_out, epochs_run * sizeof(float));
    delete[] epoch_errors_out;

    return std::make_tuple(
        new_weights_np, new_biases_np,
        py::array_t<float>(inputDim, mean_out, create_deleter(mean_out)),
        py::array_t<float>(inputDim, std_out, create_deleter(std_out)),
        py::array_t<float>(numSample * inputDim, norm_samples_out, create_deleter(norm_samples_out)),
        epoch_errors_np
    );
}


PYBIND11_MODULE(my_process_module, m) {
    m.doc() = "C++ module for perceptron algorithms";

    
    m.def("Add_Data", &PyAdd_Data_Wrapper, "Adds a new data sample");
    m.def("Add_Labels", &PyAdd_Labels_Wrapper, "Adds a new label");
    m.def("init_array_random", &PyInit_Array_Random_Wrapper, "Initializes a random float array");

    
    m.def("Calculate_Z_Score", &PyCalculate_Z_Score_Wrapper, "Calculates Z-Score parameters.");
    m.def("Predict_Binary_Class", &PyPredict_Binary_Class_Wrapper, "Predicts for binary classification.");
    m.def("Predict_Multiclass", &PyPredict_Multiclass_Wrapper, "Predicts for multiclass classification.");
    m.def("Predict_MLP_Multiclass", &PyPredict_MLP_Multiclass_Wrapper, "Predicts for MLP multiclass classification.");

    
    m.def("Train_Perceptron_Binary", &PyTrain_Perceptron_Binary_Wrapper, "Trains a binary perceptron.");
    m.def("Train_Perceptron_Multiclass", &PyTrain_Perceptron_Multiclass_Wrapper, "Trains a multiclass perceptron.");
    m.def("Train_MLP_Multiclass", &PyTrain_MLP_Multiclass_Wrapper, "Trains a flexible MLP.");
}