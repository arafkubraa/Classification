#include "Process.h"
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// Z-Score Parametrelerini hesaplar
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std) {
    float* Total = new float[dim];
    int i, j;
    for (i = 0; i < dim; i++) {
        mean[i] = std[i] = Total[i] = 0.0;
    }
    for (i = 0; i < Size; i++)
        for (j = 0; j < dim; j++)
            Total[j] += x[i * dim + j];
    for (i = 0; i < dim; i++)
        mean[i] = Total[i] / float(Size);

    for (i = 0; i < Size; i++)
        for (j = 0; j < dim; j++)
            std[j] += ((x[i * dim + j] - mean[j]) * (x[i * dim + j] - mean[j]));

    for (j = 0; j < dim; j++)
        std[j] = sqrt(std[j] / float(Size));

    delete[] Total;
}

/**
 * @brief İkili Perceptron için Tek Nokta Tahmini (Forward Pass).
 */
int Predict_Binary_Class(const float* test_sample, const float* weights, float bias, int inputDim)
{
    // Net Değerini Hesaplama (W * X + B)
    float test_net = bias;
    for (int i = 0; i < inputDim; i++) {
        test_net += weights[i] * test_sample[i];
    }
    // Sınıflandırma Mantığı: Pozitif ise sınıf 0, negatif ise sınıf 1 indeksi.
    return (test_net > 0.0f) ? 0 : 1;
}


/**
 * @brief Çoklu Perceptron için Tek Nokta Tahmini (Forward Pass).
 * En yüksek skoru üreten sınıfın indeksini döndürür.
 */
int Test_Forward_Multiclass(const float* test_sample, const float* weights, const float* bias,
                            int class_count, int inputDim)
{
    int predicted_class_idx = -1;
    float max_score = -1e9f; // Çok küçük bir başlangıç değeri

    // Her sınıf için skor hesaplaması
    for (int c = 0; c < class_count; ++c) {
        float net = bias[c];
        // Her sınıfın kendi ağırlık vektörü ile çarpım
        for (int i = 0; i < inputDim; ++i) {
            net += weights[c * inputDim + i] * test_sample[i];
        }

        if (net > max_score) {
            max_score = net;
            predicted_class_idx = c;
        }
    }
    return predicted_class_idx; // En yüksek skora sahip sınıfın indeksi 
}


// Sigmoid aktivasyon fonksiyonu
float mlp_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

/**
 * @brief Çok Katmanlı Perceptron için Tek Nokta Tahmini (Forward Pass).
 */
int Predict_MLP_Multiclass(const float* test_sample,
                           int* layer_sizes, int num_layers,
                           float** weights, float** biases)
{
    std::vector<float> current_activations(test_sample, test_sample + layer_sizes[0]);
    std::vector<float> next_activations;

    // İleri yayılım
    for (int l = 0; l < num_layers - 1; ++l) {
        int prev_layer_size = layer_sizes[l];
        int current_layer_size = layer_sizes[l + 1];
        next_activations.assign(current_layer_size, 0.0f);
        float* current_weights = weights[l];
        float* current_biases = biases[l];

        for (int j = 0; j < current_layer_size; ++j) {
            float net = current_biases[j];
            for (int i = 0; i < prev_layer_size; ++i) {
                net += current_activations[i] * current_weights[i * current_layer_size + j];
            }
            if (l < num_layers - 2) { // Gizli katmanlar için sigmoid
                next_activations[j] = mlp_sigmoid(net);
            } else { // Çıkış katmanı için lineer
                next_activations[j] = net;
            }
        }
        current_activations = next_activations;
    }

    // En yüksek skora sahip sınıfın indeksini bul
    int predicted_class_idx = -1;
    float max_score = -1e9f;
    for (int i = 0; i < layer_sizes[num_layers - 1]; ++i) {
        if (current_activations[i] > max_score) {
            max_score = current_activations[i];
            predicted_class_idx = i;
        }
    }

    return predicted_class_idx;
}