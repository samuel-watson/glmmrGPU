
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "cuda_cholesky.h"
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

GPUCholeskyManager::GPUCholeskyManager(int dim) : n(dim) {
    // Check CUDA device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found! Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nUsing device: " << prop.name << std::endl;

    matrix_size = n * n * sizeof(double);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (matrix_size > free_mem) {
        std::cout << "Need " << matrix_size / (1024 * 1024) << " MB\n";
        std::cout << "Free " << free_mem / (1024 * 1024) << " MB\n";
        std::cerr << "Not enough GPU memory!\n";
    }

    err = cudaMalloc(&d_A, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "FAILED to allocate d_matrix: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemset(d_A, 0, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        d_A = nullptr;
        return;
    }

    is_allocated = true;

    // Create cuSOLVER handle
    cusolverDnHandle_t* h_solver = new cusolverDnHandle_t;
    cusolverStatus_t solver_status = cusolverDnCreate(h_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "FAILED to create cuSOLVER handle. Status: " << solver_status << std::endl;
        cudaFree(d_A);
        delete h_solver;
        is_allocated = false;
        return;
    }
    cusolver_handle = h_solver;

    // Create cuBLAS handle
    cublasHandle_t* h_blas = new cublasHandle_t;
    cublasStatus_t blas_status = cublasCreate(h_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "FAILED to create cuBLAS handle. Status: " << blas_status << std::endl;
        cusolverDnDestroy(*h_solver);
        cudaFree(d_A);
        delete h_solver;
        delete h_blas;
        is_allocated = false;
        return;
    }
    cublas_handle = h_blas;
}

GPUCholeskyManager::~GPUCholeskyManager() {
    if (d_A) {
        cudaFree(d_A);
    }

    if (cusolver_handle) {
        cusolverDnDestroy(*static_cast<cusolverDnHandle_t*>(cusolver_handle));
        delete static_cast<cusolverDnHandle_t*>(cusolver_handle);
    }

    if (cublas_handle) {
        cublasDestroy(*static_cast<cublasHandle_t*>(cublas_handle));
        delete static_cast<cublasHandle_t*>(cublas_handle);
    }

}

void GPUCholeskyManager::upload(const Eigen::MatrixXd& A) {
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    cudaMemcpy(d_A, A.data(), matrix_size, cudaMemcpyHostToDevice);
}

void GPUCholeskyManager::compute_cholesky(const Eigen::MatrixXd& A) {
    upload(A);

    cusolverDnHandle_t handle = *static_cast<cusolverDnHandle_t*>(cusolver_handle);

    // Query workspace size for DOUBLE precision
    int lwork = 0;
    cusolverStatus_t status = cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, d_A, n, &lwork);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf_bufferSize failed with status: " << status << std::endl;
        throw std::runtime_error("Failed to query workspace size");
    }

    // Allocate workspace
    double* d_work;  // DOUBLE not float!
    cudaError_t err = cudaMalloc(&d_work, lwork * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate workspace: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Workspace allocation failed");
    }

    int* d_info;
    err = cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        throw std::runtime_error("Info allocation failed");
    }

    // Synchronize before calling Cholesky
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR BEFORE Cholesky: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Error before Cholesky");
    }

    // Compute Cholesky decomposition - DOUBLE precision
    status = cusolverDnDpotrf(  // D for double!
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,      // Device pointer
        n,        // Leading dimension (should equal n for square matrix)
        d_work,   // Workspace
        lwork,    // Workspace size
        d_info    // Info output
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf failed with status: " << status << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Synchronize after Cholesky to catch errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR DURING Cholesky execution: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky execution caused memory error");
    }

    // Check for errors
    int h_info;
    err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Info copy failed");
    }

    cudaFree(d_work);
    cudaFree(d_info);

    if (h_info != 0) {
        std::cerr << "Cholesky decomposition failed with info = " << h_info << std::endl;
        if (h_info < 0) {
            std::cerr << "The " << -h_info << "-th parameter had an illegal value" << std::endl;
        }
        else {
            std::cerr << "The leading minor of order " << h_info
                << " is not positive definite" << std::endl;
        }
        throw std::runtime_error("Matrix is not positive definite or invalid parameter");
    }
}

void GPUCholeskyManager::solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
{
    if (!is_allocated) {
        throw std::runtime_error("No L matrix to solve");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    double* d_B = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(double)));  // n x nrhs matrix
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double), cudaMemcpyHostToDevice));
    // Solve AX = B for all right-hand sides simultaneously
    CUSOLVER_CHECK(cusolverDnDpotrs(
        h,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        n,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        n,        // Leading dimension of B (number of rows)
        d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_B, n * nrhs * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
}

void GPUCholeskyManager::solve(const Eigen::MatrixXd& B, Eigen::VectorXd& X)
{
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = 1;
    double* d_B = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(double)));  // n x nrhs matrix
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double), cudaMemcpyHostToDevice));
    // Solve AX = B for all right-hand sides simultaneously
    CUSOLVER_CHECK(cusolverDnDpotrs(
        h,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        n,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        n,        // Leading dimension of B (number of rows)
        d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(X.data(), d_B, n * nrhs * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
}

void GPUCholeskyManager::solveInPlace(Eigen::MatrixXd& B)
{
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    double* d_B = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * nrhs * sizeof(double)));  // n x nrhs matrix
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double),
        cudaMemcpyHostToDevice));
    // Solve AX = B for all right-hand sides simultaneously
    CUSOLVER_CHECK(cusolverDnDpotrs(
        h,
        CUBLAS_FILL_MODE_LOWER,
        n,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        n,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        n,        // Leading dimension of B (number of rows)
        d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    // Copy solution matrix X back to host
    CUDA_CHECK(cudaMemcpy(B.data(), d_B, n * nrhs * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
}

void GPUCholeskyManager::download(Eigen::MatrixXd& X) {
    if (X.rows() != n || X.cols() != n) {
        std::cout << "Resizing Eigen matrix to " << n << "x" << n << std::endl;
        X.resize(n, n);
    }
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, d_A);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: Invalid device pointer: " << cudaGetErrorString(err) << std::endl;
        cudaGetLastError();  // Clear error
        throw std::runtime_error("Invalid device pointer");
    }
    // Check Eigen matrix data pointer
    if (X.data() == nullptr) {
        std::cerr << "ERROR: Eigen matrix data pointer is null" << std::endl;
        throw std::runtime_error("Eigen matrix not allocated");
    }

    if (X.size() * sizeof(double) != matrix_size) {
        std::cerr << "ERROR: Size mismatch!" << std::endl;
        std::cerr << "  GPU: " << matrix_size << " bytes" << std::endl;
        std::cerr << "  CPU: " << X.size() * sizeof(double) << " bytes" << std::endl;
        throw std::runtime_error("Size mismatch");
    }

    // Synchronize before copy
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaDeviceSynchronize failed: "
            << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Synchronization failed");
    }

    // Perform the copy
    err = cudaMemcpy(X.data(), d_A, matrix_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "  Source (GPU): " << d_A << std::endl;
        std::cerr << "  Dest (CPU): " << X.data() << std::endl;
        std::cerr << "  Size: " << matrix_size << " bytes" << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }
}

void GPUCholeskyManager::multiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    if (!is_allocated) {
        throw std::runtime_error("GPU memory not allocated");
    }

    const int other_cols = B.cols();
    double* d_result = nullptr;
    double* d_other = nullptr;

    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);

    const double alpha = 1.0;

    // Check for previous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Previous CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Previous CUDA error detected");
    }

    // CRITICAL: Use cublasDtrmm for DOUBLE precision (not cublasStrmm!)
    // Note: cublasDtrmm can work IN-PLACE, so we need to copy d_other to d_result first

    
    CUDA_CHECK(cudaMalloc(&d_other, n * other_cols * sizeof(double)));
    err = cudaMemcpy(d_other, B.data(), n * other_cols * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Copy failed");
    }
    CUDA_CHECK(cudaMalloc(&d_result, n * other_cols * sizeof(double)));
    

    cublasStatus_t status = cublasDtrmm(
        handle,
        CUBLAS_SIDE_LEFT,           // A (L) is on the left
        CUBLAS_FILL_MODE_LOWER,     // L is lower triangular
        CUBLAS_OP_N,                // No transpose of L
        CUBLAS_DIAG_NON_UNIT,       // Non-unit diagonal
        n,                          // m: rows of B and C
        other_cols,                 // n: cols of B and C
        &alpha,                     // scalar alpha
        d_A, n,                     // A matrix and lda
        d_other, n,                 // B matrix and ldb
        d_result, n                 // C matrix and ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDtrmm failed with status: " << status << std::endl;
        throw std::runtime_error("cublasDtrmm failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR after multiply: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Multiplication caused memory error");
    }

    CUDA_CHECK(cudaMemcpy(X.data(), d_result, n * other_cols * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_other));
}

void GPUCholeskyManager::leftMultiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    if (!is_allocated) {
        throw std::runtime_error("GPU memory not allocated");
    }

    const int other_cols = B.rows();
    double* d_result = nullptr;
    double* d_other = nullptr;

    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);

    const double alpha = 1.0;

    // Check for previous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Previous CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Previous CUDA error detected");
    }

    CUDA_CHECK(cudaMalloc(&d_other, n * other_cols * sizeof(double)));
    err = cudaMemcpy(d_other, B.data(), n * other_cols * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Copy failed");
    }
    CUDA_CHECK(cudaMalloc(&d_result, n * other_cols * sizeof(double)));


    cublasStatus_t status = cublasDtrmm(
        handle,
        CUBLAS_SIDE_RIGHT,           // A (L) is on the left
        CUBLAS_FILL_MODE_LOWER,     // L is lower triangular
        CUBLAS_OP_N,                // No transpose of L
        CUBLAS_DIAG_NON_UNIT,       // Non-unit diagonal
        other_cols,                          // m: rows of B and C
        n,                 // n: cols of B and C
        &alpha,                     // scalar alpha
        d_A, n,                     // A matrix and lda
        d_other, other_cols,                 // B matrix and ldb
        d_result, other_cols                 // C matrix and ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDtrmm failed with status: " << status << std::endl;
        throw std::runtime_error("cublasDtrmm failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR after multiply: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Multiplication caused memory error");
    }

    CUDA_CHECK(cudaMemcpy(X.data(), d_result, n * other_cols * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_other));
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

void gpu_gram_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& X) {
    // Compute C = A^T * A
    // A is m x n
    // C is n x n (symmetric)
    const int m = A.rows();
    const int n = A.cols();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    double* d_A = nullptr;
    double* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * m * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));

    double alpha = 1.0;
    double beta = 0.0;

    // C = alpha * A * A^T + beta * C
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N,    // Don't transpose first argument (A)
        CUBLAS_OP_T,    // Transpose second argument (A^T)
        m,              // Rows of result C
        m,              // Columns of result C
        n,              // Inner dimension
        &alpha,
        d_A,            // First matrix A
        m,              // Leading dimension of A
        d_A,            // Second matrix A (will be transposed)
        m,              // Leading dimension of A
        &beta,
        d_C,            // Result matrix C
        m               // Leading dimension of C
    ));

    CUDA_CHECK(cudaMemcpy(X.data(), d_C, m * m * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));


}

