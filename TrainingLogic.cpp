#include "Process.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

void Train_Perceptron_Binary_C(float* samples, float* targets,
                               float* weights, float* bias,
                               int numSample, int inputDim,
                               float* mean_out, float* std_out,
                               float* normalized_samples_out,
                               float* epoch_errors_out, int* epochs_run_out)
{
    Z_Score_Parameters(samples, numSample, inputDim, mean_out, std_out);

    float* adjustedTargets = new float[numSample];

    for (int i = 0; i < numSample; ++i) {
        for (int d = 0; d < inputDim; ++d) {
            float std_val = (std_out[d] != 0.0f) ? std_out[d] : 1.0f;
            normalized_samples_out[i * inputDim + d] = (samples[i * inputDim + d] - mean_out[d]) / std_val;
        }
        adjustedTargets[i] = (targets[i] == 1.0f) ? 1.0f : -1.0f;
    }

    float total_err, net, out;
    float lrn_cnst = 0.001f;
    int max_iter = 10000;
    int epoch = 0;
    bool learned = false;
    float bias_copy = bias[0];

    float momentum_coeff = 0.9f;
    std::vector<float> prev_weight_update(inputDim, 0.0f);
    float prev_bias_update = 0.0f;

    while (!learned && epoch < max_iter) {
        total_err = 0.0f;
        epoch++;

        for (int k = 0; k < numSample; k++) {
            const float* current_sample = normalized_samples_out + k * inputDim;
            net = bias_copy;
            for (int i = 0; i < inputDim; i++) {
                net += weights[i] * current_sample[i];
            }
            out = (net > 0.0f) ? 1.0f : -1.0f;
            float err = adjustedTargets[k] - out;

            if (std::fabs(err) > 1e-6f) {
                for (int i = 0; i < inputDim; i++) {
                    float current_update = lrn_cnst * (err / 2.0f) * current_sample[i];
                    float momentum_update = current_update + momentum_coeff * prev_weight_update[i];
                    weights[i] += momentum_update;
                    prev_weight_update[i] = momentum_update;
                }
                float current_bias_update = lrn_cnst * (err / 2.0f);
                float momentum_bias_update = current_bias_update + momentum_coeff * prev_bias_update;
                bias_copy += momentum_bias_update;
                prev_bias_update = momentum_bias_update;
            }
            total_err += std::fabs(err);
        }
        epoch_errors_out[epoch - 1] = total_err;
        if (total_err < 1e-6f) {
            learned = true;
        }
    }
    *epochs_run_out = epoch;
    bias[0] = bias_copy;
    delete[] adjustedTargets;
}


void Train_Perceptron_Multiclass_C(float* samples, float* targets,
                                   float* weights, float* bias,
                                   int numSample, int inputDim, int class_count,
                                   float* mean_out, float* std_out,
                                   float* normalized_samples_out,
                                   float* epoch_errors_out, int* epochs_run_out)
{
    Z_Score_Parameters(samples, numSample, inputDim, mean_out, std_out);

    for (int i = 0; i < numSample; ++i) {
        for (int d = 0; d < inputDim; ++d) {
            float std_val = (std_out[d] != 0.0f) ? std_out[d] : 1.0f;
            normalized_samples_out[i * inputDim + d] = (samples[i * inputDim + d] - mean_out[d]) / std_val;
        }
    }

    float total_err;
    float lrn_cnst = 0.001f;
    int max_iter = 10000;
    int epoch = 0;
    bool learned = false;
    std::vector<float> scores(class_count);

    float momentum_coeff = 0.9f;
    std::vector<float> prev_weight_update((size_t)class_count * inputDim, 0.0f);
    std::vector<float> prev_bias_update((size_t)class_count, 0.0f);

    while (!learned && epoch < max_iter) {
        total_err = 0.0f;
        epoch++;

        for (int k = 0; k < numSample; k++) {
            const float* current_sample = normalized_samples_out + k * inputDim;
            int true_label_idx = static_cast<int>(targets[k]) - 1;
            int predicted_class_idx = -1;
            float max_score = -1e9f;

            for (int c = 0; c < class_count; ++c) {
                float net = bias[c];
                for (int i = 0; i < inputDim; ++i) {
                    net += weights[c * inputDim + i] * current_sample[i];
                }
                scores[c] = net;
                if (net > max_score) {
                    max_score = net;
                    predicted_class_idx = c;
                }
            }

            if (predicted_class_idx != true_label_idx) {
                total_err += 1.0f;
                for (int i = 0; i < inputDim; i++) {
                    int w_idx = predicted_class_idx * inputDim + i;
                    float current_update = -lrn_cnst * current_sample[i];
                    float momentum_update = current_update + momentum_coeff * prev_weight_update[w_idx];
                    weights[w_idx] += momentum_update;
                    prev_weight_update[w_idx] = momentum_update;
                }
                float current_bias_update = -lrn_cnst;
                float momentum_bias_update = current_bias_update + momentum_coeff * prev_bias_update[predicted_class_idx];
                bias[predicted_class_idx] += momentum_bias_update;
                prev_bias_update[predicted_class_idx] = momentum_bias_update;

                for (int i = 0; i < inputDim; i++) {
                    int w_idx = true_label_idx * inputDim + i;
                    float current_update = lrn_cnst * current_sample[i];
                    float momentum_update = current_update + momentum_coeff * prev_weight_update[w_idx];
                    weights[w_idx] += momentum_update;
                    prev_weight_update[w_idx] = momentum_update;
                }
                current_bias_update = lrn_cnst;
                momentum_bias_update = current_bias_update + momentum_coeff * prev_bias_update[true_label_idx];
                bias[true_label_idx] += momentum_bias_update;
                prev_bias_update[true_label_idx] = momentum_bias_update;
            }
        }
        epoch_errors_out[epoch - 1] = total_err;
        if (total_err < 1e-6f) {
            learned = true;
        }
    }
    *epochs_run_out = epoch;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float sigmoid_val) {
    return sigmoid_val * (1.0f - sigmoid_val);
}

void Train_MLP_Multiclass_C(float* samples, float* targets,
                            int numSample,
                            int* layer_sizes, int num_layers,
                            float** weights, float** biases,
                            float* mean_out, float* std_out,
                            float* normalized_samples_out,
                            float* epoch_errors_out, int* epochs_run_out)
{
    int inputDim = layer_sizes[0];
    int outputDim = layer_sizes[num_layers - 1];

    //Normalizasyon
    Z_Score_Parameters(samples, numSample, inputDim, mean_out, std_out);
    for (int i = 0; i < numSample; ++i) {
        for (int d = 0; d < inputDim; ++d) {
            float std_val = (std_out[d] != 0.0f) ? std_out[d] : 1.0f;
            normalized_samples_out[i * inputDim + d] = (samples[i * inputDim + d] - mean_out[d]) / std_val;
        }
    }

    float lrn_cnst = 0.001f;
    float momentum_coeff = 0.9f;
    int max_iter = 10000;

    // Bellek Ayırma
    std::vector<std::vector<float>> activations(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        activations[i].resize(layer_sizes[i]);
    }

    std::vector<std::vector<float>> deltas(num_layers - 1);
    for (int i = 0; i < num_layers - 1; ++i) {
        deltas[i].resize(layer_sizes[i + 1]);
    }

    std::vector<std::vector<float>> prev_weight_updates;
    for (int i = 0; i < num_layers - 1; ++i) {
        prev_weight_updates.emplace_back(layer_sizes[i] * layer_sizes[i + 1], 0.0f);
    }

    std::vector<std::vector<float>> prev_bias_updates;
    for (int i = 0; i < num_layers - 1; ++i) {
        prev_bias_updates.emplace_back(layer_sizes[i + 1], 0.0f);
    }

    int epoch = 0;
    bool learned = false;
    while (!learned && epoch < max_iter) {
        float total_err = 0.0f;
        epoch++;

        for (int k = 0; k < numSample; k++) {
            const float* current_sample = normalized_samples_out + k * inputDim;
            std::copy(current_sample, current_sample + inputDim, activations[0].begin());
            int true_label_idx = static_cast<int>(targets[k]) - 1;

            //İleri Yayılım Kısmı
            for (int l = 0; l < num_layers - 1; ++l) {
                int prev_layer_size = layer_sizes[l];
                int current_layer_size = layer_sizes[l + 1];
                float* current_weights = weights[l];
                float* current_biases = biases[l];
                
                for (int j = 0; j < current_layer_size; ++j) {
                    float net = current_biases[j];
                    for (int i = 0; i < prev_layer_size; ++i) {
                        net += activations[l][i] * current_weights[i * current_layer_size + j];
                    }
                    if (l < num_layers - 2) { // Gizli katmanlar
                        activations[l + 1][j] = sigmoid(net);
                    } else { // Çıkış katmanı
                        activations[l + 1][j] = net;
                    }
                }
            }
            
            // Softmax çıkış katmanında
            float max_val = *std::max_element(activations.back().begin(), activations.back().end());
            float sum_exp = 0.0f;
            for(int i = 0; i < outputDim; ++i) {
                activations.back()[i] = exp(activations.back()[i] - max_val);
                sum_exp += activations.back()[i];
            }
            for(int i = 0; i < outputDim; ++i) {
                activations.back()[i] /= sum_exp;
            }

            total_err += -log(activations.back()[true_label_idx] + 1e-9f);

            // Geri Yayılım Kısmı
            // Çıkış Katmanı
            for (int j = 0; j < outputDim; ++j) {
                float target_val = (j == true_label_idx) ? 1.0f : 0.0f;
                deltas.back()[j] = activations.back()[j] - target_val;
            }

            // Gizli Katman
            for (int l = num_layers - 2; l > 0; --l) {
                int next_layer_size = layer_sizes[l + 1];
                int current_layer_size = layer_sizes[l];
                float* next_layer_weights = weights[l];
                
                for (int j = 0; j < current_layer_size; ++j) {
                    float error = 0.0f;
                    for (int m = 0; m < next_layer_size; ++m) {
                        error += deltas[l][m] * next_layer_weights[j * next_layer_size + m];
                    }
                    deltas[l-1][j] = error * sigmoid_derivative(activations[l][j]);
                }
            }

            // Ağırlık güncelleme
            for (int l = num_layers - 2; l >= 0; --l) {
                int prev_layer_size = layer_sizes[l];
                int current_layer_size = layer_sizes[l + 1];
                float* current_weights = weights[l];
                float* current_biases = biases[l];

                for (int j = 0; j < current_layer_size; ++j) {
                    for (int i = 0; i < prev_layer_size; ++i) {
                        int w_idx = i * current_layer_size + j;
                        float current_update = -lrn_cnst * deltas[l][j] * activations[l][i];
                        float momentum_update = current_update + momentum_coeff * prev_weight_updates[l][w_idx];
                        current_weights[w_idx] += momentum_update;
                        prev_weight_updates[l][w_idx] = momentum_update;
                    }
                    float current_bias_update = -lrn_cnst * deltas[l][j];
                    float momentum_bias_update = current_bias_update + momentum_coeff * prev_bias_updates[l][j];
                    current_biases[j] += momentum_bias_update;
                    prev_bias_updates[l][j] = momentum_bias_update;
                }
            }
        } //sample döngüsü


        epoch_errors_out[epoch - 1] = total_err / numSample;
        if ((total_err / numSample) < 1e-4f) {
            learned = true;
        }
    } // epoch döngüsü

    *epochs_run_out = epoch;
}