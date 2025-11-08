
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "cuda_cholesky.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>

__global__ void addToDiagonal(double* matrix, int n, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        matrix[idx * n + idx] += value;  // Access diagonal element [i,i]
    }
}

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

    if (3 * matrix_size > free_mem) {
        std::cout << "Need " << matrix_size / (1024 * 1024) << " MB\n";
        std::cout << "Free " << free_mem / (1024 * 1024) << " MB\n";
        std::cerr << "Not enough GPU memory!\n";
    }

    cudaMallocHost(&h_pinned_A, matrix_size);
    cudaMallocHost(&h_pinned_B, matrix_size);

    err = cudaMalloc(&d_A, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "FAILED to allocate d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_B, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "FAILED to allocate d_B: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_C, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "FAILED to allocate d_C: " << cudaGetErrorString(err) << std::endl;
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

    // Allocate workspace ONCE
    cusolverDnDpotrf_bufferSize(*static_cast<cusolverDnHandle_t*>(cusolver_handle), CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, &workspace_size);
    cudaMalloc(&d_work, workspace_size * sizeof(double));
    cudaMalloc(&d_info, sizeof(int));
    
}

GPUCholeskyManager::~GPUCholeskyManager() {
    if (d_A) {
        cudaFree(d_A);
    }

    if (d_B) {
        cudaFree(d_B);
    }

    if (d_C) {
        cudaFree(d_C);
    }

    if (cusolver_handle) {
        cusolverDnDestroy(*static_cast<cusolverDnHandle_t*>(cusolver_handle));
        delete static_cast<cusolverDnHandle_t*>(cusolver_handle);
    }

    if (cublas_handle) {
        cublasDestroy(*static_cast<cublasHandle_t*>(cublas_handle));
        delete static_cast<cublasHandle_t*>(cublas_handle);
    }

    if (h_pinned_A) {
        cudaFreeHost(h_pinned_A);
    }
    if (h_pinned_B) {
        cudaFreeHost(h_pinned_B);
    }

    cudaFree(d_work);
    cudaFree(d_info);
}

void GPUCholeskyManager::upload(const Eigen::MatrixXd& A) {
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    std::memcpy(h_pinned_A, A.data(), matrix_size);
    cudaMemcpy(d_A, h_pinned_A, matrix_size, cudaMemcpyHostToDevice);
}

void GPUCholeskyManager::computeCholesky(const Eigen::MatrixXd& A) {
    upload(A);
    cusolverDnHandle_t handle = *static_cast<cusolverDnHandle_t*>(cusolver_handle);    

    cusolverStatus_t status = cusolverDnDpotrf(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,      // Device pointer
        n,        // Leading dimension (should equal n for square matrix)
        d_work,   // Workspace
        workspace_size,    // Workspace size
        d_info    // Info output
    );
    

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf failed with status: " << status << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Check for errors
    int h_info;
    cudaError_t err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Info copy failed");
    }

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

void GPUCholeskyManager::computeAndSolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    upload(A);
    cusolverDnHandle_t handle = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    size_t b_size = n * nrhs * sizeof(double);
    std::memcpy(h_pinned_A, A.data(), matrix_size);
    CUDA_CHECK(cudaMemcpy(d_A, h_pinned_A, matrix_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream1)
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream2)
    //cudaDeviceSynchronize();

    cusolverStatus_t status = cusolverDnDpotrf(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,      // Device pointer
        n,        // Leading dimension (should equal n for square matrix)
        d_work,   // Workspace
        workspace_size,    // Workspace size
        d_info    // Info output
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf failed with status: " << status << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Check for errors
    int h_info;
    cudaError_t err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Info copy failed");
    }

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

    CUSOLVER_CHECK(cusolverDnDpotrs(
        handle, CUBLAS_FILL_MODE_LOWER, n, nrhs,     
        d_A, n, d_B, n, d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, b_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, b_size);

}

void GPUCholeskyManager::multCompSolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C, Eigen::MatrixXd& X) {
    if (A.rows() > n || A.cols() > n || B.rows() > n || B.cols() > n) throw std::runtime_error("Max dim. for multiplyBuffer is n");
    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    cusolverDnHandle_t handles = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int m = A.rows();
    const int k = A.cols();
    const int l = B.cols();
    const int nrhs = C.cols();
    size_t a_size = m * k * sizeof(double);
    size_t b_size = k * l * sizeof(double);
    size_t c_size = m * l * sizeof(double);
    size_t d_size = m * nrhs * sizeof(double);
    std::memcpy(h_pinned_A, A.data(), a_size);
    CUDA_CHECK(cudaMemcpy(d_C, h_pinned_A, a_size, cudaMemcpyHostToDevice));
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));
    double alpha = 1.0f;
    double beta = 0.0f;
    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, m, l, k,
        &alpha,d_C, m,d_B, k,&beta, d_A, m));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    ::addToDiagonal<<<numBlocks, blockSize>>>(d_A, n, 1.0);

    cusolverStatus_t status = cusolverDnDpotrf(
        handles, CUBLAS_FILL_MODE_LOWER, n, d_A, n,     
        d_work, workspace_size, d_info    
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf failed with status: " << status << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Check for errors
    int h_info;
    cudaError_t err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Info copy failed");
    }

    std::memcpy(h_pinned_A, C.data(), d_size);
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_pinned_A, d_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    CUSOLVER_CHECK(cusolverDnDpotrs(
        handles, CUBLAS_FILL_MODE_LOWER, n, nrhs, d_A,      
        n, d_B,  n, d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, d_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, d_size);
}

void GPUCholeskyManager::multCompSolve2(const Eigen::MatrixXd& A, const Eigen::VectorXd& W, const Eigen::VectorXd& C, Eigen::MatrixXd& X) {
    if (A.rows() > n || A.cols() > n || W.size() > n) throw std::runtime_error("Max dim. for multiplyBuffer is n");
    
    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    cusolverDnHandle_t handles = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int m = A.rows();
    const int k = A.cols();
    const int nrhs = C.cols();
    size_t a_size = m * k * sizeof(double);
    size_t w_size = m * sizeof(double);
    size_t d_size = m * sizeof(double);
    std::memcpy(h_pinned_A, A.data(), a_size);
    CUDA_CHECK(cudaMemcpy(d_C, h_pinned_A, a_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_A, a_size, cudaMemcpyHostToDevice));
    std::memcpy(h_pinned_B, W.data(), w_size);
    CUDA_CHECK(cudaMemcpy(d_A, h_pinned_B, w_size, cudaMemcpyHostToDevice));

    // W * X
    CUBLAS_CHECK(cublasDdgmm(
        handle, CUBLAS_SIDE_LEFT, m,k, d_C, m, d_A, 1,  
        d_C,  m           
    ));

    double alpha = 1.0;
    double beta = 0.0;

    // X^T * W * X
    CUBLAS_CHECK(cublasDgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,k, k, 
        m, &alpha,d_B, m, d_C,m, &beta,d_A, k   
    ));

    // X^T * W * X + 1
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    ::addToDiagonal<<<numBlocks, blockSize>>>(d_A, n, 1.0);

    // cholesky factorisation
    cusolverStatus_t status = cusolverDnDpotrf(
        handles,CUBLAS_FILL_MODE_LOWER,n,d_A, n,       
        d_work,  workspace_size, d_info   
    );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDpotrf failed with status: " << status << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Cholesky decomposition failed");
    }

    // Check for errors
    int h_info;
    cudaError_t err = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy info: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_work);
        cudaFree(d_info);
        throw std::runtime_error("Info copy failed");
    }

    std::memcpy(h_pinned_A, C.data(), d_size);
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_pinned_A, d_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    double* d_result;
    cudaMalloc(&d_result, d_size);

    // W^T X^T y
    CUBLAS_CHECK(cublasDgemv(
        handle, CUBLAS_OP_T,  m,  k,&alpha,       
        d_C, m, d_B,1, &beta,d_result,1                
    ));

    // (X^T * W * X + 1)^-1 W^T X^T y
    CUSOLVER_CHECK(cusolverDnDpotrs(
        handles, CUBLAS_FILL_MODE_LOWER,n,1,     
        d_A, n, d_result, n, d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_result, d_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, d_size);
}

void GPUCholeskyManager::solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
{
    if (!is_allocated) {
        throw std::runtime_error("No L matrix to solve");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    size_t b_size = n * nrhs * sizeof(double);
    //CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * nrhs * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

   

    CUSOLVER_CHECK(cusolverDnDpotrs(
        h, CUBLAS_FILL_MODE_LOWER, n,
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

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, b_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, b_size);
}

void GPUCholeskyManager::solveInPlace(Eigen::MatrixXd& B)
{
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    size_t b_size = n * nrhs * sizeof(double);

    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));
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

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, b_size, cudaMemcpyDeviceToHost));
    std::memcpy(B.data(), h_pinned_B, b_size);

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
    err = cudaMemcpy(h_pinned_A, d_A, matrix_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed");
    }
    std::memcpy(X.data(), h_pinned_A, matrix_size);
}

void GPUCholeskyManager::multiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    if (!is_allocated) {
        throw std::runtime_error("GPU memory not allocated");
    }

    const int other_cols = B.cols();
    if (other_cols > n) throw std::runtime_error("Too many columns for GPU multiplication in B");

    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    const double alpha = 1.0;
    
    size_t b_size = n * other_cols * sizeof(double);
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasDtrmm(
        handle,  CUBLAS_SIDE_LEFT,  CUBLAS_FILL_MODE_LOWER,     
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,       
        n,  other_cols,  &alpha, d_A, n,                    
        d_B, n,  d_C, n                 
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDtrmm failed with status: " << status << std::endl;
        throw std::runtime_error("cublasDtrmm failed");
    }

    cudaMemcpy(h_pinned_B, d_C, b_size, cudaMemcpyDeviceToHost);
    std::memcpy(X.data(), h_pinned_B, b_size);
 }

void GPUCholeskyManager::leftMultiplyByMatrix(const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    if (!is_allocated) {
        throw std::runtime_error("GPU memory not allocated");
    }

    const int other_cols = B.rows();
    if (other_cols > n) throw std::runtime_error("Too many columns for GPU multiplication in B");
    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    const double alpha = 1.0;

    // Check for previous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Previous CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Previous CUDA error detected");
    }

    size_t b_size = n * other_cols * sizeof(double);
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasDtrmm(
        handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,    
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, other_cols,                          
        n, &alpha,  d_A, n,d_B, other_cols,d_C, other_cols                
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDtrmm failed with status: " << status << std::endl;
        throw std::runtime_error("cublasDtrmm failed");
    }

    //CUDA_CHECK(cudaMemcpy(X.data(), d_result, n * other_cols * sizeof(double), cudaMemcpyDeviceToHost));
    cudaMemcpy(h_pinned_B, d_C, b_size, cudaMemcpyDeviceToHost);
    std::memcpy(X.data(), h_pinned_B, b_size);
}

void GPUCholeskyManager::multiplyBuffer(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X) {
    
    if (A.rows() > n || A.cols() > n || B.rows() > n || B.cols() > n) throw std::runtime_error("Max dim. for multiplyBuffer is n");
    double* d_result = nullptr;
    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);

    const int m = A.rows();
    const int k = A.cols();
    const int l = B.cols();
    size_t a_size = m * k * sizeof(double);
    size_t b_size = k * l * sizeof(double);
    size_t c_size = m * l * sizeof(double);
    std::memcpy(h_pinned_A, A.data(), a_size);
    CUDA_CHECK(cudaMemcpy(d_A, h_pinned_A, a_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream1)
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream2)

    //cudaDeviceSynchronize();

    double alpha = 1.0f;
    double beta = 0.0f;

    CUBLAS_CHECK(cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, l, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m));

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_C, c_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, c_size);
}





