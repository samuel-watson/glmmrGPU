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
class RandomEffects{
public:
  MatrixXd    u_;
  MatrixXd    scaled_u_;
  MatrixXd    zu_;
  modeltype&  model;
  int         mcmc_block_size = 1; // for saem
  
  RandomEffects(modeltype& model_) : 
    u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    scaled_u_(MatrixXd::Zero(model_.covariance.Q(),1)),
    zu_(model_.n(),1), model(model_) {};
  
  RandomEffects(modeltype& model_, int n, int Q) : 
    u_(MatrixXd::Zero(Q,1)),
    scaled_u_(MatrixXd::Zero(Q,1)),
    zu_(n,1), model(model_) {};
  
  RandomEffects(const glmmr::RandomEffects<modeltype>& re) : u_(re.u_), scaled_u_(re.scaled_u_), zu_(re.zu_), model(re.model) {};
  
  MatrixXd      Zu(){return zu_;};
  MatrixXd      u(bool scaled = true);
  VectorMatrix  predict_re(const ArrayXXd& newdata_);
  
};

}

template<typename modeltype>
inline MatrixXd glmmr::RandomEffects<modeltype>::u(bool scaled){
  if(scaled){
    return model.covariance.Lu(u_);
  } else {
    return u_;
  }
}

template<>
inline VectorMatrix glmmr::RandomEffects<bits>::predict_re(const ArrayXXd& newdata_){
  if(model.covariance.data_.cols()!=newdata_.cols())throw std::runtime_error("Different numbers of columns in new data");
  // generate the merged data
  int nnew = newdata_.rows();
  ArrayXXd mergedata(model.n()+nnew,model.covariance.data_.cols());
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
  MatrixXd D = covariancenew.D(false,false);
  result.mat = D.block(model.covariance.Q(),model.covariance.Q(),newQ,newQ);
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

template<>
inline VectorMatrix glmmr::RandomEffects<bits_hsgp>::predict_re(const ArrayXXd& newdata_){
  if(model.covariance.data_.cols()!=newdata_.cols())throw std::runtime_error("Different numbers of columns in new data");
  
  hsgpCovariance covariancenewnew(model.covariance.form_,
                                  newdata_,
                                  model.covariance.colnames_);
  
  covariancenewnew.update_parameters(model.covariance.parameters_);
  MatrixXd newLu = covariancenewnew.Lu(u(false));
  int iter = newLu.cols();
  
  // //generate sigma
  int newQ = newdata_.rows();//covariancenewnew.Q();
  VectorMatrix result(newQ);
  result.vec.setZero();
  result.mat.setZero();
  result.vec = newLu.rowwise().mean();
  VectorXd newLuCol(newLu.rows());
  for(int i = 0; i < iter; i++){
    newLuCol = newLu.col(i) - result.vec;
    result.mat += (newLuCol * newLuCol.transpose());
  }
  result.mat.array() *= (1/(double)iter);
  return result;
}