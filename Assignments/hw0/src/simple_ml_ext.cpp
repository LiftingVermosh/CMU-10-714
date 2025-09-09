#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * 对数据执行单轮SGD训练实现softmax回归（C++版本）。
     * 按批次顺序处理数据，并原地更新theta矩阵。
     *
     * 参数：
     *   X: 输入数据指针，形状为(m, n)
     *   y: 标签指针，形状为(m,)
     *   theta: 参数矩阵指针，形状为(n, k)，会被原地修改
     *   m: 样本数量
     *   n: 特征维度
     *   k: 类别数量
     *   lr: 学习率
     *   batch: 批处理大小
     */
    // 循环处理每个批次
    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);
        size_t current_batch_size = end - start;

        // 临时存储当前批次的梯度，初始化为0
        std::vector<float> grad(n * k, 0.0f);

        // 处理当前批次中的每个样本
        for (size_t i = start; i < end; ++i) {
            // 计算当前样本的logits Z: 1xk向量
            std::vector<float> Z_row(k, 0.0f);
            float max_Z = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < k; ++j) {
                for (size_t l = 0; l < n; ++l) {
                    Z_row[j] += X[i * n + l] * theta[l * k + j];
                }
                if (Z_row[j] > max_Z) {
                    max_Z = Z_row[j];
                }
            }

            // 计算exp(Z - max_Z)用于数值稳定性
            std::vector<float> exp_Z_row(k);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                exp_Z_row[j] = std::exp(Z_row[j] - max_Z);
                sum_exp += exp_Z_row[j];
            }

            // 计算softmax概率
            std::vector<float> probs(k);
            for (size_t j = 0; j < k; ++j) {
                probs[j] = exp_Z_row[j] / sum_exp;
            }

            // 构建当前样本的one-hot编码标签
            std::vector<unsigned char> one_hot(k, 0);
            one_hot[y[i]] = 1;

            // 计算当前样本的梯度贡献并累加到grad中
            for (size_t j = 0; j < k; ++j) {
                float diff = probs[j] - one_hot[j];
                for (size_t l = 0; l < n; ++l) {
                    grad[l * k + j] += diff * X[i * n + l];
                }
            }
        }

        // 更新theta：使用平均梯度（除以当前批次大小）和学习率
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                theta[i * k + j] -= lr * grad[i * k + j] / current_batch_size;
            }
        }
    }
}

PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
            softmax_regression_epoch_cpp(
                static_cast<const float*>(X.request().ptr),
                static_cast<const unsigned char*>(y.request().ptr),
                static_cast<float*>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch
            );
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}


// 优化版本 By Deepseek
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <cmath>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <limits>

// namespace py = pybind11;

// void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
//                                   float *theta, size_t m, size_t n, size_t k,
//                                   float lr, size_t batch) {
//     /**
//      * 使用C++实现的softmax回归单轮训练
//      * 优化点：
//      *   - 预分配临时数组，避免循环内动态内存分配
//      *   - 优化循环顺序（先特征后类别）以改善缓存性能
//      *   - 重用内存，减少开销
//      * 
//      * 参数：
//      *   X: 输入数据指针，形状为(m, n)
//      *   y: 标签指针，形状为(m,)
//      *   theta: 参数矩阵指针，形状为(n, k)，会被原地修改
//      *   m: 样本数量
//      *   n: 特征维度
//      *   k: 类别数量
//      *   lr: 学习率
//      *   batch: 批处理大小
//      */
//     // 预分配临时数组，避免循环内动态内存分配
//     std::vector<float> grad(n * k);       // 梯度矩阵，大小n*k
//     std::vector<float> Z_row(k);          // 单个样本的logits，大小k
//     std::vector<float> exp_Z_row(k);      // 用于数值稳定性的exp(Z - max_Z)，大小k
//     std::vector<float> probs(k);          // softmax概率，大小k
//     std::vector<unsigned char> one_hot(k); // one-hot编码标签，大小k

//     // 循环处理每个批次
//     for (size_t start = 0; start < m; start += batch) {
//         size_t end = std::min(start + batch, m);
//         size_t current_batch_size = end - start;

//         // 重置当前批次的梯度为零
//         std::fill(grad.begin(), grad.end(), 0.0f);

//         // 处理当前批次中的每个样本
//         for (size_t i = start; i < end; ++i) {
//             // 重置当前样本的logits为零
//             std::fill(Z_row.begin(), Z_row.end(), 0.0f);

//             // 计算样本i的logits Z_row，优化循环顺序：先特征(l)后类别(j)
//             for (size_t l = 0; l < n; ++l) {
//                 float x_il = X[i * n + l];  // 特征值(i, l)
//                 for (size_t j = 0; j < k; ++j) {
//                     Z_row[j] += x_il * theta[l * k + j];  // 累加类别j的logit
//                 }
//             }

//             // 查找最大logit值用于数值稳定性
//             float max_Z = -std::numeric_limits<float>::infinity();
//             for (size_t j = 0; j < k; ++j) {
//                 if (Z_row[j] > max_Z) {
//                     max_Z = Z_row[j];
//                 }
//             }

//             // 计算exp(Z - max_Z)及其总和用于softmax
//             float sum_exp = 0.0f;
//             for (size_t j = 0; j < k; ++j) {
//                 exp_Z_row[j] = std::exp(Z_row[j] - max_Z);
//                 sum_exp += exp_Z_row[j];
//             }

//             // 计算softmax概率
//             for (size_t j = 0; j < k; ++j) {
//                 probs[j] = exp_Z_row[j] / sum_exp;
//             }

//             // 重置one_hot为零并设置真实标签
//             std::fill(one_hot.begin(), one_hot.end(), 0);
//             one_hot[y[i]] = 1;

//             // 计算梯度贡献，优化循环顺序：先特征(l)后类别(j)
//             for (size_t l = 0; l < n; ++l) {
//                 float x_il = X[i * n + l];
//                 for (size_t j = 0; j < k; ++j) {
//                     grad[l * k + j] += (probs[j] - one_hot[j]) * x_il;
//                 }
//             }
//         }

//         // 使用平均梯度（除以批次大小）和学习率更新theta
//         for (size_t i = 0; i < n; ++i) {
//             for (size_t j = 0; j < k; ++j) {
//                 theta[i * k + j] -= lr * grad[i * k + j] / current_batch_size;
//             }
//         }
//     }
// }

// // Pybind11模块定义，用于Python调用
// PYBIND11_MODULE(simple_ml_ext, m) {
//     m.def("softmax_regression_epoch_cpp",
//         [](py::array_t<float, py::array::c_style> X,
//            py::array_t<unsigned char, py::array::c_style> y,
//            py::array_t<float, py::array::c_style> theta,
//            float lr,
//            int batch) {
//             // 调用C++实现的softmax回归训练函数
//             softmax_regression_epoch_cpp(
//                 static_cast<const float*>(X.request().ptr),
//                 static_cast<const unsigned char*>(y.request().ptr),
//                 static_cast<float*>(theta.request().ptr),
//                 X.request().shape[0],
//                 X.request().shape[1],
//                 theta.request().shape[1],
//                 lr,
//                 batch
//             );
//         },
//         py::arg("X"), py::arg("y"), py::arg("theta"),
//         py::arg("lr"), py::arg("batch"));
// }
