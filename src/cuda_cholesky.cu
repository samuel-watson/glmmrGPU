
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSolver error: " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while(0)

void gpu_chol_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    // Initialize cuSolver
    const int n = A.rows();
    const int nrhs = B.cols();
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    double* d_A = nullptr;
    double* d_B = nullptr;  // This will store B as input and X as output
    int* d_info = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(double)));  // n x nrhs matrix
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double),
        cudaMemcpyHostToDevice));

    // Query workspace size and allocate
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

    // Perform Cholesky factorization
    std::cout << "Performing Cholesky factorization..." << std::endl;
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, d_work, lwork, d_info));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Cholesky factorization failed!" << std::endl;
    }

    // Solve AX = B for all right-hand sides simultaneously
    std::cout << "Solving AX = B for all " << nrhs << " right-hand sides..." << std::endl;
    CUSOLVER_CHECK(cusolverDnDpotrs(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        n,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        n,        // Leading dimension of B (number of rows)
        d_info
    ));

    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_B, n * nrhs * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

}

void gpu_chol(const Eigen::MatrixXd& A, Eigen::MatrixXd& X) {
    // Initialize cuSolver
    const int n = A.rows();
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    double* d_A = nullptr;
    int* d_info = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double),
        cudaMemcpyHostToDevice));

    // Query workspace size and allocate
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

    // Perform Cholesky factorization
    std::cout << "Performing Cholesky factorization..." << std::endl;
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, d_work, lwork, d_info));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Cholesky factorization failed!" << std::endl;
    }

    
    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_A, n * n * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

}

void gpu_chol_solve_existing(const double* A_data, const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    // Initialize cuSolver
    const int n = B.rows();
    const int nrhs = B.cols();
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    double* d_A = nullptr;
    double* d_B = nullptr;  // This will store B as input and X as output
    int* d_info = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(double)));  // n x nrhs matrix
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_data, n * n * sizeof(double),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double),
        cudaMemcpyHostToDevice));

    // Query workspace size and allocate
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

    // Perform Cholesky factorization
    std::cout << "Performing Cholesky factorization..." << std::endl;
    CUSOLVER_CHECK(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, d_work, lwork, d_info));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Cholesky factorization failed!" << std::endl;
    }

    // Solve AX = B for all right-hand sides simultaneously
    std::cout << "Solving AX = B for all " << nrhs << " right-hand sides..." << std::endl;
    CUSOLVER_CHECK(cusolverDnDpotrs(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        n,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        n,        // Leading dimension of B (number of rows)
        d_info
    ));

    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_B, n * nrhs * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

}

void gpu_multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    // Initialize cuSolver
    const int m = A.rows();
    const int k = A.cols();
    const int n = B.cols();
    
    // Allocate device memory
    int* d_info = nullptr;

    double * d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), m * k * sizeof(double),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), k * n * sizeof(double),
        cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major order (like Eigen's default)
    double alpha = 1.0f;
    double beta = 0.0f;

    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_C, m * n * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

}

