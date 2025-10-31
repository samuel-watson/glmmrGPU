#pragma once

#include <Eigen/Dense>

void gpu_chol_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);

void gpu_chol(const Eigen::MatrixXd& A, Eigen::MatrixXd& X);

void gpu_chol_solve_existing(const double* A_data, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);

void gpu_multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);

void gpu_gram_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& X);

void gpu_gram_cholesky(const Eigen::MatrixXd& A, Eigen::MatrixXd& L, int m, int n);