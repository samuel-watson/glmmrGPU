#pragma once

#include <Eigen/Dense>

class GPUCholeskyManager {
public:
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    void* cusolver_handle;
    void* cublas_handle;
    int n;
    size_t matrix_size;
    int workspace_size;
    bool is_allocated = false;

    GPUCholeskyManager(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager& operator=(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager(int dim);
    ~GPUCholeskyManager();

    void upload(const Eigen::MatrixXd& A);
    void compute_cholesky(const Eigen::MatrixXd& A);
    void solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void solve(const Eigen::MatrixXd& B, Eigen::VectorXd& X);
    void solveInPlace(Eigen::MatrixXd& B);
    void download(Eigen::MatrixXd& X);
    void multiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void leftMultiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
};

void gpu_multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);

void gpu_gram_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& X);
