#pragma once

#include "covariance.hpp"
#include "modelbits.hpp"

namespace glmmr {

    using namespace Eigen;

    enum class RandomEffectMargin {
        AtEstimated = 0,
        At = 1,
        AtZero = 2,
        Average = 3
    };


    template<typename modeltype>
    class RandomEffects {
    public:
        MatrixXd    u_;
        MatrixXd    scaled_u_;
        MatrixXd    zu_;
        VectorXd    u_mean_;
        MatrixXd    u_solve_;
        ArrayXd     u_weight_;
        VectorXd    u_loglik_;
        modeltype& model;
        int         mcmc_block_size = 1; // for saem

        RandomEffects(modeltype& model_) :
            u_(MatrixXd::Zero(model_.covariance.Q(), 1)),
            scaled_u_(MatrixXd::Zero(model_.covariance.Q(), 1)),
            zu_(model_.n(), 1), u_mean_(VectorXd::Zero(model_.covariance.Q())),
            u_solve_(MatrixXd::Zero(model_.covariance.Q(), 1)),
            u_weight_(VectorXd::Zero(1)), u_loglik_(VectorXd::Zero(1)), model(model_) {
        };

        RandomEffects(modeltype& model_, int n, int Q) :
            u_(MatrixXd::Zero(Q, 1)),
            scaled_u_(MatrixXd::Zero(Q, 1)),
            zu_(n, 1), u_mean_(Q), u_solve_(Q, 1),
            u_weight_(1), u_loglik_(1), model(model_) {
        };

        RandomEffects(const glmmr::RandomEffects<modeltype>& re) : u_(re.u_), scaled_u_(re.scaled_u_),
            zu_(re.zu_), u_mean_(re.u_mean_), u_solve_(re.u_solve_), u_weight_(re.u_weight_),
            u_loglik_(re.u_loglik_), model(re.model) {
        };

        MatrixXd      Zu() { return zu_; };
        MatrixXd      u(bool scaled = true);
        VectorMatrix  predict_re(const ArrayXXd& newdata_);
        void          update_zu(const bool weights);
    };

}

template<typename modeltype>
inline MatrixXd glmmr::RandomEffects<modeltype>::u(bool scaled) {
    if (scaled) {
        return model.covariance.Lu(u_);
    }
    else {
        return u_;
    }
}

template<typename modeltype>
inline void glmmr::RandomEffects<modeltype>::update_zu(const bool weights) {
    scaled_u_ = model.covariance.Lu(u_);
    MatrixXd Z = model.covariance.Z();
    zu_ = Z * scaled_u_;
    ArrayXd xb = model.xb();
    model.covariance.matL.solve(scaled_u_, u_solve_);
    u_weight_.setZero();
    if (weights) {
#pragma omp parallel for 
        for (int i = 0; i < scaled_u_.cols(); i++) {
            double llmod = maths::log_likelihood(model.data.y.array(), xb + zu_.col(i).array(), model.data.variance, model.family);
            double llprior = -0.5 * scaled_u_.col(i).dot(u_solve_.col(i));
            u_weight_(i) = llmod + llprior - u_loglik_(i);
        }
        u_weight_ -= u_weight_.maxCoeff();
        u_weight_ = u_weight_.exp();
        double weightsum = u_weight_.sum();
        u_weight_ *= 1.0 / weightsum;
    }
    else {
        u_weight_.setConstant(1.0 / scaled_u_.cols());
    }
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits>::predict_re(const ArrayXXd& newdata_) {
    if (model.covariance.data_.cols() != newdata_.cols())throw std::runtime_error("Different numbers of columns in new data");
    // generate the merged data
    int nnew = newdata_.rows();
    ArrayXXd mergedata(model.n() + nnew, model.covariance.data_.cols());
    mergedata.topRows(model.n()) = model.covariance.data_;
    mergedata.bottomRows(nnew) = newdata_;

    Covariance covariancenew(model.covariance.form_,
        mergedata,
        model.covariance.colnames_,
        NO_MATL);

    covariancenew.update_parameters(model.covariance.parameters_);
    int newQ = covariancenew.Q() - model.covariance.Q();
    std::cout << "\nNew Q: " << newQ << std::endl;
    VectorMatrix result(newQ);
    MatrixXd D = covariancenew.D(false, false);
    result.mat = D.block(model.covariance.Q(), model.covariance.Q(), newQ, newQ);
    MatrixXd DU(model.covariance.Q(), u_.cols());
    MatrixXd DD(model.covariance.Q(), newQ);
    MatrixXd Lu = model.covariance.Lu(u(false));
    MatrixXd D12 = D.block(model.covariance.Q(), 0, newQ, model.covariance.Q());
    model.covariance.matL.solve(Lu, DU);
    model.covariance.matL.solve(D12.transpose(), DD);
    MatrixXd SSV = D12 * DU;
    result.vec = SSV.rowwise().mean();
    result.mat -= D12 * DD;
    return result;
}

