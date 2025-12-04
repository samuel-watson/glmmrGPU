#pragma once

//defines

#define _USE_MATH_DEFINES
#include <omp.h> 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Sparse>

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS 
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

//glmmrbase version 
#define GLMMR10
#define GLMMR11
#define GLMMR12

// includes

#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <set>
#include <map>
#include <unordered_map>
#include <deque>

using namespace Eigen;

typedef std::string str;
typedef std::vector<str> strvec;
typedef std::vector<int> intvec;
typedef std::vector<double> dblvec;
typedef std::vector<strvec> strvec2d;
typedef std::vector<dblvec> dblvec2d;
typedef std::vector<intvec> intvec2d;
typedef std::vector<dblvec2d> dblvec3d;
typedef std::vector<intvec2d> intvec3d;
typedef std::pair<double, double> dblpair;
typedef std::pair<std::string, double> strdblpair;



