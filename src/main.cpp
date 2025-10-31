#include <iostream>
#include <chrono>
#include <fstream>
#include <Eigen/Dense>
#include <filesystem>
#include <map>
#include "cuda_cholesky.h"
#include "model.hpp"

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template <typename Derived>
void writeToCSVfile(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

Eigen::MatrixXd readMatrixFromCSV(std::string file, const int rows, const int cols) {
    std::ifstream in(file);

    std::string line;
    int row = 0;
    int col = 0;
    Eigen::MatrixXd res(rows, cols);

    if (in.is_open()) {
        while (std::getline(in, line)) {
            char* ptr = (char*)line.c_str();
            int len = line.length();

            col = 0;
            char* start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == ',') {
                    res(row, col++) = atof(start);
                    start = ptr + i + 1;
                }
            }
            res(row, col) = atof(start);
            row++;
        }
        in.close();
    }
    return res;
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    size_t end = str.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

std::map<std::string, std::string> readConfig(const std::string& filename) {
    std::ifstream file(filename);
    std::map<std::string, std::string> config;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {

        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            config[key] = value;
        }
    }

    return config;
}

std::vector<std::string> splitString(const std::string& str, char delimiter = ' ') {
    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string token;

    while (iss >> token) {  // This automatically handles spaces
        result.push_back(token);
    }

    return result;
}

std::vector<double> splitStringD(const std::string& str, char delimiter = ' ') {
    std::vector<double> result;
    std::istringstream iss(str);
    std::string token;

    while (iss >> token) {  // This automatically handles spaces
        result.push_back(std::stod(token));
    }

    return result;
}

int main() {
    using namespace Eigen;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    typedef glmmr::Model<glmmr::ModelBits<glmmr::Covariance, glmmr::LinearPredictor> > glmm;
    // read in the data 
#ifdef _DEBUG
    auto config = readConfig("C:/Users/samue/source/repos/glmmrGPU/model.txt");
#else
    auto config = readConfig("model.txt");  // For release builds
#endif
    std::string form = config["formula"];
    
    int nrows = std::stoi(config["nrows"]);
    int ncols = std::stoi(config["ncols"]);
    int niter = std::stoi(config["niter"]);
    std::string family = config["family"];
    std::string link = config["link"];
    std::vector<std::string> columns = splitString(config["colnames"]);
    std::vector<double> start_cov = splitStringD(config["covariance"]);
    std::vector<double> start_b = splitStringD(config["mean"]);
    //formula

    std::cout << "\nFormula: " << form << " nrows " << nrows << " ncols " << ncols;
    
    // data
#ifdef _DEBUG
    ArrayXXd data = (readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/X.csv", nrows, ncols)).array();
    MatrixXd ymat = readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/y.csv", nrows, 1);
#else
    ArrayXXd data = (readMatrixFromCSV("data/data.csv", nrows, ncols)).array();
    MatrixXd ymat = readMatrixFromCSV("data/y.csv", nrows, 1);
#endif
    
    VectorXd y = Map<VectorXd>(ymat.data(), ymat.rows());

    std::cout << "\nData:\n" << data.topLeftCorner(10, 3);
    std::cout << "\ny:\n" << y.head(10).transpose();
    
    glmm model(form, data, columns, family, link);
    model.model.linear_predictor.update_parameters(start_b);
    model.model.covariance.update_parameters(start_cov);
    model.set_y(y);

    std::cout << "\nD:\n" << model.model.covariance.D().topLeftCorner(5, 5);
    
    // let's do one iteration
    auto t1 = high_resolution_clock::now();
    model.matrix.posterior_u_samples(niter, 1e-6, false);
    auto t2 = high_resolution_clock::now();

    std::cout << "\n U: \n" << model.re.u_.topLeftCorner(5, 5);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "\nTiming: " << ms_double.count() << "ms\n";
    
    /*
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
    */
    return 0;
}