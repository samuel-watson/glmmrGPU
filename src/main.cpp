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

    if (!in.is_open()) {
        std::cerr << "ERROR: Could not open file: " << file << std::endl;
        return MatrixXd::Zero(1,1);
    }

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
    /*
#ifdef _DEBUG
    auto config = readConfig("C:/Users/samue/source/repos/glmmrGPU/model.txt");
#else
    auto config = readConfig("model.txt");  // For release builds
#endif*/
    auto config = readConfig("C:/Users/samue/source/repos/glmmrGPU/model.txt");
    std::string form = config["formula"];
    
    int nrows = std::stoi(config["nrows"]);
    int ncols = std::stoi(config["ncols"]);
    int niter = std::stoi(config["niter"]);
    int maxiter = std::stoi(config["maxiter"]);
    int predict = std::stoi(config["predict"]);
    int offset = std::stoi(config["offset"]);
    std::string family = config["family"];
    std::string link = config["link"];
    std::vector<std::string> columns = splitString(config["colnames"]);
    std::vector<double> start_cov = splitStringD(config["covariance"]);
    std::vector<double> start_b = splitStringD(config["mean"]);
    //formula

    std::cout << "\nFormula: " << form << " nrows " << nrows << " ncols " << ncols;
    
    // data
    /*
#ifdef _DEBUG
    ArrayXXd data = (readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/X.csv", nrows, ncols)).array();
    MatrixXd ymat = readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/y.csv", nrows, 1);
#else
    ArrayXXd data = (readMatrixFromCSV("data/data.csv", nrows, ncols)).array();
    MatrixXd ymat = readMatrixFromCSV("data/y.csv", nrows, 1);
#endif*/

    ArrayXXd data = (readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/X.csv", nrows, ncols)).array();
    MatrixXd ymat = readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/y.csv", nrows, 1);
    
    
    VectorXd y = Map<VectorXd>(ymat.data(), ymat.rows());
        
    glmm model(form, data, columns, family, link);
    model.model.linear_predictor.update_parameters(start_b);
    model.model.covariance.update_parameters(start_cov);
    model.set_y(y);

    MatrixXd offs(1, 1);
    ArrayXXd trials(1, 1);
    if (offset) {
        offs.resize(data.rows(), NoChange);
        offs = readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/offset.csv", nrows, 1);
        model.set_offset(offs.col(0));
    }
    if (family == "binomial") {
        trials.resize(data.rows(), NoChange);
        trials = readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/trials.csv", nrows, 1);
        model.model.data.set_variance(trials.col(0));
    }
    // let's do one iteration
    auto t1 = high_resolution_clock::now();
    model.fit(niter, maxiter);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "\nTiming: " << ms_double.count() << "ms\n";

    if (predict) {
        int nrowpredict = std::stoi(config["nrowspred"]);
        ArrayXXd pred_data = (readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/Xp.csv", nrowpredict, ncols)).array();
        ArrayXXd offset_new(nrowpredict, 1);
        if (offset) {
            offset_new = (readMatrixFromCSV("C:/Users/samue/source/repos/glmmrGPU/offsetp.csv", nrowpredict, 1)).array();
        }
        else {
            offset_new.setZero();
        }
        VectorMatrix rep = model.re.predict_re(pred_data);
        VectorXd xb = model.model.linear_predictor.predict_xb(pred_data, offset_new);
        writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/pred_u_mean.csv", rep.vec);
        writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/pred_u_var.csv", rep.mat);
        writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/pred_xb.csv", xb);
    }

    // write results to file
    dblvec beta = model.model.linear_predictor.parameters;
    dblvec theta = model.model.covariance.parameters_;
    double var_par = model.model.data.var_par;
    MatrixXd M(beta.size(), beta.size());
    MatrixXd Mt(theta.size(), theta.size());
    int se = std::stoi(config["se"]);
    if (se) {
        M = model.matrix.information_matrix();
        Mt = model.matrix.template information_matrix_theta<glmmr::IM::EIM>();
        M = M.llt().solve(MatrixXd::Identity(M.rows(), M.cols()));
        Mt = Mt.llt().solve(MatrixXd::Identity(Mt.rows(), Mt.cols()));
    }
    else {
        M.setZero();
        Mt.setZero();
    }
    
    int dim = beta.size() + theta.size();
    if (family == "gaussian")dim++;
    MatrixXd result(dim, 2);
    for (int i = 0; i < beta.size(); i++) {
        result(i, 0) = beta[i];
        result(i, 1) = sqrt(M(i, i));
    }
    for (int i = 0; i < theta.size(); i++) {
        result(i + beta.size(), 0) = theta[i];
        result(i + beta.size(), 1) = sqrt(Mt(i, i));
    }
    if (family == "gaussian") {
        result(beta.size() + theta.size(), 0) = var_par;
        result(beta.size() + theta.size(), 1) = 0;
    }

    // need to add predict mode and return of U samples

    writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/result.csv", result);
    writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/u.csv", model.re.zu_);
/*
#ifdef _DEBUG
    writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/result.csv", result);
    writeToCSVfile("C:/Users/samue/source/repos/glmmrGPU/u.csv", model.re.zu_);
#else
    writeToCSVfile("result.csv", result);
    writeToCSVfile("u.csv", model.re.zu_);
#endif*/
    
   
    return 0;
}