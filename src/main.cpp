#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "cuda_cholesky.h"

int main() {
    using namespace Eigen;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    const int n = 100;      // Matrix dimension
    const int nrhs = n;

    //MatrixXd m = MatrixXd::Random(n, n);
    //MatrixXd A = m * m.transpose();
    //MatrixXd B = MatrixXd::Identity(n, n);
    MatrixXd A = MatrixXd::Random(n, n);
    MatrixXd B = MatrixXd::Random(n, n);

    // Verify with Eigen
    auto t1 = high_resolution_clock::now();
    MatrixXd X_expected = A * B;// A.llt().solve(B);
    auto t2 = high_resolution_clock::now();
    std::cout << "Expected solution X (from Eigen):\n" << X_expected.topLeftCorner(5, 5) << "\n" << std::endl;

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "s\n";

    MatrixXd X(n, n);

    auto t3 = high_resolution_clock::now();
    gpu_multiply(A, B, X);//gpu_chol(A, B, X);
    auto t4 = high_resolution_clock::now();

    std::cout << "Actual solution X\n" << X.topLeftCorner(5, 5) << "\n" << std::endl;
    duration<double, std::milli> ms_double2 = t4 - t3;
    std::cout << ms_double2.count() << "ms\n";

    std::cout << "\nDifference from expected solution (Frobenius norm): "
        << (X - X_expected).norm() << std::endl;

    return 0;
}