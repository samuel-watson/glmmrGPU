
#include <iostream>
#include <chrono>
#include <vector>
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

__global__ void scaleRowsBySqrtW(double* WZ, const double* Z, const double* W, int n, int q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * q) {
        int row = idx % n;  // Column-major
        WZ[idx] = sqrt(W[row]) * Z[idx];
    }
}

__global__ void scaleVectorByWInPlace(double* v, const double* W, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = W[idx] * v[idx];
    }
}

__global__ void addIdentityToDiagonal(double* A, int q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < q) {
        A[idx * q + idx] += 1.0;
    }
}

__global__ void columnwiseDotProduct(double* out, const double* A, const double* B,
    int dim, int niter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < niter) {
        double sum = 0.0;
        for (int d = 0; d < dim; d++) {
            int idx = d + col * dim;  // Column-major
            sum += A[idx] * B[idx];
        }
        out[col] = sum;
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

GPUCholeskyManager::GPUCholeskyManager(int dim) : Tn(dim) {
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

    matrix_size = Tn * Tn * sizeof(double);
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
        Tn, d_A, Tn, &workspace_size);
    cudaMalloc(&d_work, workspace_size * sizeof(double));
    cudaMalloc(&d_info, sizeof(int));
    
}

GPUCholeskyManager::GPUCholeskyManager(){}

GPUCholeskyManager::~GPUCholeskyManager() {
    clearRandomEffectsSolve();

    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    if (cusolver_handle) {
        cusolverDnDestroy(*static_cast<cusolverDnHandle_t*>(cusolver_handle));
        delete static_cast<cusolverDnHandle_t*>(cusolver_handle);
    }

    if (cublas_handle) {
        cublasDestroy(*static_cast<cublasHandle_t*>(cublas_handle));
        delete static_cast<cublasHandle_t*>(cublas_handle);
    }

    if (h_pinned_A) cudaFreeHost(h_pinned_A);
    if (h_pinned_B)cudaFreeHost(h_pinned_B);
    if(d_work)cudaFree(d_work);
    if(d_info)cudaFree(d_info);
}

void GPUCholeskyManager::upload(const Eigen::MatrixXd& A) {
    if (!is_allocated) {
        throw std::runtime_error("No allocated memory");
    }
    Ln = A.rows();
    if (Ln > Tn) throw std::runtime_error("Ln > n");
    matrix_size = Ln * Ln * sizeof(double);
    std::memcpy(h_pinned_A, A.data(), matrix_size);
    cudaMemcpy(d_A, h_pinned_A, matrix_size, cudaMemcpyHostToDevice);
}

void GPUCholeskyManager::computeCholesky(const Eigen::MatrixXd& A) {
    upload(A);
    cusolverDnHandle_t handle = *static_cast<cusolverDnHandle_t*>(cusolver_handle);    

    cusolverStatus_t status = cusolverDnDpotrf(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        Ln,
        d_A,      // Device pointer
        Ln,        // Leading dimension (should equal n for square matrix)
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
    size_t b_size = Ln * nrhs * sizeof(double);

    std::memcpy(h_pinned_A, A.data(), matrix_size);
    CUDA_CHECK(cudaMemcpy(d_A, h_pinned_A, matrix_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream1)
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));//, *static_cast<cudaStream_t*>(stream2)
    //cudaDeviceSynchronize();

    cusolverStatus_t status = cusolverDnDpotrf(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        Ln,
        d_A,      // Device pointer
        Ln,        // Leading dimension (should equal n for square matrix)
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
        handle, CUBLAS_FILL_MODE_LOWER, Ln, nrhs,     
        d_A, Ln, d_B, Ln, d_info
    ));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        std::cerr << "Linear solve failed!" << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, b_size, cudaMemcpyDeviceToHost));
    std::memcpy(X.data(), h_pinned_B, b_size);

}

void GPUCholeskyManager::solve(const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
{
    if (!is_allocated) {
        throw std::runtime_error("No L matrix to solve");
    }
    cusolverDnHandle_t h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    const int nrhs = B.cols();
    size_t b_size = Ln * nrhs * sizeof(double);
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnDpotrs(
        h, CUBLAS_FILL_MODE_LOWER, Ln,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        Ln,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        Ln,        // Leading dimension of B (number of rows)
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
    size_t b_size = Ln * nrhs * sizeof(double);

    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));
    // Solve AX = B for all right-hand sides simultaneously
    CUSOLVER_CHECK(cusolverDnDpotrs(
        h,
        CUBLAS_FILL_MODE_LOWER,
        Ln,
        nrhs,     // Number of columns in B (and X)
        d_A,      // Factorized matrix
        Ln,        // Leading dimension of A
        d_B,      // Input: B matrix, Output: X matrix
        Ln,        // Leading dimension of B (number of rows)
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
    if (X.rows() != Ln || X.cols() != Ln) {
        std::cout << "Resizing Eigen matrix to " << Ln << "x" << Ln << std::endl;
        X.resize(Ln, Ln);
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
    if (other_cols > Tn) throw std::runtime_error("Too many columns for GPU multiplication in B");

    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    const double alpha = 1.0;
    
    size_t b_size = Ln * other_cols * sizeof(double);
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasDtrmm(
        handle,  CUBLAS_SIDE_LEFT,  CUBLAS_FILL_MODE_LOWER,     
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,       
        Ln,  other_cols,  &alpha, d_A, Ln,                    
        d_B, Ln,  d_C, Ln                 
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
    if (other_cols > Tn) throw std::runtime_error("Too many columns for GPU multiplication in B");
    cublasHandle_t handle = *static_cast<cublasHandle_t*>(cublas_handle);
    const double alpha = 1.0;

    // Check for previous errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Previous CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Previous CUDA error detected");
    }

    size_t b_size = Ln * other_cols * sizeof(double);
    std::memcpy(h_pinned_B, B.data(), b_size);
    CUDA_CHECK(cudaMemcpy(d_B, h_pinned_B, b_size, cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasDtrmm(
        handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,    
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, other_cols,                          
        Ln, &alpha,  d_A, Ln,d_B, other_cols,d_C, other_cols                
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
    
    if (A.rows() > Tn || A.cols() > Tn || B.rows() > Tn || B.cols() > Tn) throw std::runtime_error("Max dim. for multiplyBuffer is n");
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
void GPUCholeskyManager::initRandomEffectsSolve(const Eigen::MatrixXd& Z) {
    if (re_allocated) {
        clearRandomEffectsSolve();
    }

    re_n = Z.rows();
    re_q = Z.cols();

    if (re_n > Tn || re_q > Tn) {
        throw std::runtime_error("Z dimensions exceed allocated buffer size");
    }

    // Store Z in d_C
    size_t z_size = re_n * re_q * sizeof(double);
    std::memcpy(h_pinned_A, Z.data(), z_size);
    CUDA_CHECK(cudaMemcpy(d_C, h_pinned_A, z_size, cudaMemcpyHostToDevice));

    // Allocate only vector buffers
    CUDA_CHECK(cudaMalloc(&d_re_W, re_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_re_vec, re_n * sizeof(double)));

    re_allocated = true;
}
void GPUCholeskyManager::solveRandomEffects(
    const Eigen::VectorXd& W,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& r,
    Eigen::VectorXd& result)
{
    if (!re_allocated) {
        throw std::runtime_error("Random effects solve not initialised");
    }
    if (!is_allocated) {
        throw std::runtime_error("GPU memory not allocated");
    }

    result.resize(re_q);

    cusolverDnHandle_t solver_h = *static_cast<cusolverDnHandle_t*>(cusolver_handle);
    cublasHandle_t blas_h = *static_cast<cublasHandle_t*>(cublas_handle);

    const int blockSize = 256;
    int numBlocks;

    // ============================================
    // Step 1: Copy W to device
    // ============================================
    std::memcpy(h_pinned_B, W.data(), re_n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_re_W, h_pinned_B, re_n * sizeof(double), cudaMemcpyHostToDevice));

    // ============================================
    // Step 2: Compute Z^T W Z using dsyrk
    // d_C holds Z, d_B will hold sqrt(W)*Z
    // ============================================
    numBlocks = (re_n * re_q + blockSize - 1) / blockSize;
    scaleRowsBySqrtW <<<numBlocks, blockSize >>> (d_B, d_C, d_re_W, re_n, re_q);
    CUDA_CHECK(cudaGetLastError());

    // Z^T W Z = (sqrt(W)*Z)^T * (sqrt(W)*Z) -> d_A
    double alpha = 1.0, beta = 0.0;
    CUBLAS_CHECK(cublasDsyrk(blas_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
        re_q, re_n, &alpha, d_B, re_n, &beta, d_A, re_q));

    // Add identity to d_A
    numBlocks = (re_q + blockSize - 1) / blockSize;
    addIdentityToDiagonal <<<numBlocks, blockSize >>> (d_A, re_q);
    CUDA_CHECK(cudaGetLastError());

    // ============================================
    // Step 3: Compute RHS = Z^T W u + Z^T r
    // ============================================

    // Copy u to d_re_vec, compute W.*u
    std::memcpy(h_pinned_B, u.data(), re_n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_re_vec, h_pinned_B, re_n * sizeof(double), cudaMemcpyHostToDevice));

    numBlocks = (re_n + blockSize - 1) / blockSize;
    scaleVectorByWInPlace <<<numBlocks, blockSize >>> (d_re_vec, d_re_W, re_n);
    CUDA_CHECK(cudaGetLastError());

    // Z^T * (W.*u) -> d_B (as RHS vector)
    alpha = 1.0; beta = 0.0;
    CUBLAS_CHECK(cublasDgemv(blas_h, CUBLAS_OP_T, re_n, re_q,
        &alpha, d_C, re_n, d_re_vec, 1, &beta, d_B, 1));

    // Copy r to d_re_vec, compute Z^T * r and add to d_B
    std::memcpy(h_pinned_B, r.data(), re_n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_re_vec, h_pinned_B, re_n * sizeof(double), cudaMemcpyHostToDevice));

    alpha = 1.0; beta = 1.0;
    CUBLAS_CHECK(cublasDgemv(blas_h, CUBLAS_OP_T, re_n, re_q,
        &alpha, d_C, re_n, d_re_vec, 1, &beta, d_B, 1));

    // ============================================
    // Step 4: Solve (Z^T W Z + I) x = rhs via Cholesky
    // System matrix in d_A, RHS in d_B
    // Cholesky factor remains in d_A for later use
    // ============================================
    Ln = re_q;  // Update Ln so solveInPlace works correctly

    CUSOLVER_CHECK(cusolverDnDpotrf(solver_h, CUBLAS_FILL_MODE_LOWER,
        re_q, d_A, re_q, d_work, workspace_size, d_info));

    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("Cholesky factorisation failed in solveRandomEffects, info = "
            + std::to_string(h_info));
    }

    CUSOLVER_CHECK(cusolverDnDpotrs(solver_h, CUBLAS_FILL_MODE_LOWER,
        re_q, 1, d_A, re_q, d_B, re_q, d_info));

    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("Cholesky solve failed in solveRandomEffects");
    }

    // ============================================
    // Step 5: Copy result back
    // ============================================
    CUDA_CHECK(cudaMemcpy(h_pinned_B, d_B, re_q * sizeof(double), cudaMemcpyDeviceToHost));
    std::memcpy(result.data(), h_pinned_B, re_q * sizeof(double));
}

void GPUCholeskyManager::clearRandomEffectsSolve() {
    if (d_re_W) { cudaFree(d_re_W); d_re_W = nullptr; }
    if (d_re_vec) { cudaFree(d_re_vec); d_re_vec = nullptr; }
    re_n = 0;
    re_q = 0;
    re_allocated = false;
}

void GPUCholeskyManager::computeGradientHessian(
    const std::vector<Eigen::MatrixXd>& S,
    const Eigen::MatrixXd& vmat,
    const Eigen::MatrixXd& umat,
    const Eigen::ArrayXd& uweight,
    Eigen::ArrayXd& logl,
    Eigen::VectorXd& grad,
    Eigen::MatrixXd& M)
{
    const int npars = S.size();
    const int dim = vmat.rows();
    const int niter = vmat.cols();

    if (dim > Tn) {
        throw std::runtime_error("Matrix dimension exceeds buffer size");
    }

    cublasHandle_t blas_h = *static_cast<cublasHandle_t*>(cublas_handle);
    const int blockSize = 256;
    int numBlocks = (niter + blockSize - 1) / blockSize;

    size_t mat_size = dim * niter * sizeof(double);
    size_t S_size = dim * dim * sizeof(double);

    // Allocate working memory on GPU
    double* d_vmat, * d_umat, * d_quadforms;
    CUDA_CHECK(cudaMalloc(&d_vmat, mat_size));
    CUDA_CHECK(cudaMalloc(&d_umat, mat_size));
    CUDA_CHECK(cudaMalloc(&d_quadforms, niter * sizeof(double)));

    // Upload vmat and umat
    std::memcpy(h_pinned_A, vmat.data(), mat_size);
    CUDA_CHECK(cudaMemcpy(d_vmat, h_pinned_A, mat_size, cudaMemcpyHostToDevice));

    std::memcpy(h_pinned_A, umat.data(), mat_size);
    CUDA_CHECK(cudaMemcpy(d_umat, h_pinned_A, mat_size, cudaMemcpyHostToDevice));

    // ============================================
    // Log-likelihood quadratic form: logl += -0.5 * vmat.col(i).dot(umat.col(i))
    // ============================================
    columnwiseDotProduct <<<numBlocks, blockSize >>> (d_quadforms, d_vmat, d_umat, dim, niter);
    CUDA_CHECK(cudaGetLastError());

    Eigen::VectorXd qf(niter);
    CUDA_CHECK(cudaMemcpy(qf.data(), d_quadforms, niter * sizeof(double), cudaMemcpyDeviceToHost));
    logl += -0.5 * qf.array();

    // ============================================
    // Compute Sv[j] = S[j] * vmat, Su[j] = S[j] * umat
    // Gradient computed on the fly
    // ============================================
    std::vector<Eigen::MatrixXd> Sv(npars), Su(npars);
    double alpha = 1.0, beta = 0.0;

    for (int j = 0; j < npars; j++) {
        // Upload S[j] to d_B
        std::memcpy(h_pinned_A, S[j].data(), S_size);
        CUDA_CHECK(cudaMemcpy(d_B, h_pinned_A, S_size, cudaMemcpyHostToDevice));

        // Sv[j] = S[j] * vmat -> d_C
        CUBLAS_CHECK(cublasDgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, niter, dim,
            &alpha, d_B, dim, d_vmat, dim,
            &beta, d_C, dim));

        // Download Sv[j]
        Sv[j].resize(dim, niter);
        CUDA_CHECK(cudaMemcpy(h_pinned_B, d_C, mat_size, cudaMemcpyDeviceToHost));
        std::memcpy(Sv[j].data(), h_pinned_B, mat_size);

        // Gradient: grad(j) += 0.5 * sum_i uweight(i) * umat.col(i).dot(Sv[j].col(i))
        columnwiseDotProduct <<<numBlocks, blockSize >>> (d_quadforms, d_umat, d_C, dim, niter);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(qf.data(), d_quadforms, niter * sizeof(double), cudaMemcpyDeviceToHost));
        grad(j) += 0.5 * (uweight * qf.array()).sum();

        // Su[j] = S[j] * umat -> d_C
        CUBLAS_CHECK(cublasDgemm(blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, niter, dim,
            &alpha, d_B, dim, d_umat, dim,
            &beta, d_C, dim));

        // Download Su[j]
        Su[j].resize(dim, niter);
        CUDA_CHECK(cudaMemcpy(h_pinned_B, d_C, mat_size, cudaMemcpyDeviceToHost));
        std::memcpy(Su[j].data(), h_pinned_B, mat_size);
    }

    // ============================================
    // Hessian: M(j,k) += -0.5*trace(S[j]*S[k]) + uweight.dot(Su[j] .* Sv[k])
    // ============================================
    for (int j = 0; j < npars; j++) {
        for (int k = j; k < npars; k++) {
            // Trace: trace(S[j] * S[k]) = sum of element-wise product
            double tr = (S[j].array() * S[k].array()).sum();
            M(j, k) += -0.5 * tr;

            // Quadratic forms
            Eigen::VectorXd quadforms = (Su[j].array() * Sv[k].array()).colwise().sum().transpose();
            M(j, k) += (uweight * quadforms.array()).sum();

            if (j != k) {
                M(k, j) = M(j, k);
            }
        }
    }

    // Cleanup
    cudaFree(d_vmat);
    cudaFree(d_umat);
    cudaFree(d_quadforms);
}