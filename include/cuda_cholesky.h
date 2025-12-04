#pragma once

#include <Eigen/Dense>

class GPUCholeskyManager {
public:
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    double* d_work = nullptr;
    double* d_re_W = nullptr;     // re_n - diagonal weights
    double* d_re_vec = nullptr;   // re_n - working vector for u, r
    int* d_info = nullptr;
    void* cusolver_handle;
    void* cublas_handle;
    int Tn; // total sample size
    int Ln;// A matrix dimension
    size_t matrix_size;
    int workspace_size;
    bool is_allocated = false;
    double* h_pinned_A;  // Pinned buffer for matrix A
    double* h_pinned_B;  // Pinned buffer for matrix B/results
    int re_n = 0;               // observations dimension (captured at init)
    int re_q = 0;               // random effects dimension (captured at init)
    bool re_allocated = false;


    GPUCholeskyManager(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager& operator=(const GPUCholeskyManager&) = delete;
    GPUCholeskyManager(int dim);
    GPUCholeskyManager();
    ~GPUCholeskyManager();

    void upload(const Eigen::MatrixXd& A);
    void computeCholesky(const Eigen::MatrixXd& A);
    void solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    //void solveAndMultiplyTr(const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, double& tr_val, Eigen::MatrixXd& X, Eigen::MatrixXd& Y);
    void computeAndSolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void solveInPlace(Eigen::MatrixXd& B);
    void download(Eigen::MatrixXd& X);
    void multiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void leftMultiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    void multiplyBuffer(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);
    // Random effects methods
    void initRandomEffectsSolve(const Eigen::MatrixXd& Z);
    void solveRandomEffects(const Eigen::VectorXd& W, const Eigen::VectorXd& u,
        const Eigen::VectorXd& r, Eigen::VectorXd& result);
    void clearRandomEffectsSolve();
    void computeGradientHessian(
        const std::vector<Eigen::MatrixXd>& S,
        const Eigen::MatrixXd& vmat,
        const Eigen::MatrixXd& umat,
        const Eigen::ArrayXd& uweight,
        Eigen::ArrayXd& logl,
        Eigen::VectorXd& grad,
        Eigen::MatrixXd& M);
};

