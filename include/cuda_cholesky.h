#pragma once

#include <Eigen/Dense>

class GPUCholeskyManager {
public:
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    void* cusolver_handle;
    void* cublas_handle;
    int n;
    size_t matrix_size;
    int workspace_size;
    bool is_allocated = false;
    double* h_pinned_A;  // Pinned buffer for matrix A
    double* h_pinned_B;  // Pinned buffer for matrix B/results

    GPUCholeskyManager(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager& operator=(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager(int dim);
    ~GPUCholeskyManager();

    void upload(const Eigen::MatrixXd& A);
    void computeCholesky(const Eigen::MatrixXd& A);
    void solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void computeAndSolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void solveInPlace(Eigen::MatrixXd& B);
    void multCompSolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, Eigen::MatrixXd& X);
    void multCompSolve2(const Eigen::MatrixXd& A, const Eigen::VectorXd& W, const Eigen::VectorXd& C, Eigen::MatrixXd& X);
    void download(Eigen::MatrixXd& X);
    void multiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void leftMultiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void multiplyBuffer(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
};

