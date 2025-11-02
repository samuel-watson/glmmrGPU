# glmmrGPU - GPU-Accelerated Mixed Model Fitting

GPU-accelerated mixed model fitting using CUDA, cuSolver, and Eigen.

## Prerequisites

Before building, ensure you have the following installed:

### Required Software

1. (Windows) **Visual Studio 2019 or 2022** (Community Edition is free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select "Desktop development with C++"
   - Make sure to include the latest Windows SDK

2. **CUDA Toolkit 11.0 or later**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Recommended: CUDA 12.x

3. **CMake 3.18 or later**
   - Download: https://cmake.org/download/

4. **Git** (optional, for cloning)
   - Download: https://git-scm.com/downloads
   - Or download project as ZIP from repository

### Hardware Requirements

- NVIDIA GPU with compute capability 5.2 or higher

## Quick Start -- Windows

### 1. Get the Code

**Option A: Clone with Git**
```powershell
git clone https://github.com/samuel-watson/glmmrGPU.git
cd glmmrGPU
```

**Option B: Download ZIP**
- Download ZIP from repository
- Extract to a folder (e.g., `C:\Projects\glmmrGPU`)
- Open PowerShell in that folder

### 2. Build the Project
Note that the compilation will download Eigen and required Boost libraries. The Boost math libraries are relatively large and downloading may take some time.

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64

# Build Release version
cmake --build . --config Release

# Or build Debug version
cmake --build . --config Debug
```

**Note:** If using Visual Studio 2019, replace with:
```powershell
cmake .. -G "Visual Studio 16 2019" -A x64
```
### 3. Add Data
The program requires data files to be in the directory of the executable. At a minimum these are model.txt, X.csv, and y.csv, which are described below. See the files in /data for an example.

### 4. Run the Program
```powershell
# From the build directory
.\Release\glmmrGPU.exe 

# Or with absolute path
.\Release\glmmrGPU.exe 
```

## Building on Linux

### Standard Build
```bash
# From project root
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Build Types

**Release (optimized):**
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Debug (with debug symbols):**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

### Run the program

Linux:
```bash
./glmmrGPU 
./glmmrGPU 
```

## Project Structure
```
glmmrGPU/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/
│   └── cuda_cholesky.h     
├── src/
│   ├── cuda_cholesky.cu    # CUDA implementation
│   └── main.cpp            # Main program
├── data/                   # Input data files
│   ├── model.txt           # Example data files
│   ├── X.csv
│   └── y.csv
└── build/                  # Build output (generated)
    ├── Debug/
    │   └── glmmrGPU.exe
    └── Release/
        └── glmmrGPU.exe
```

## Data
The model data should be passed to the program through files placed in the folder with the program. 

### model.txt
```
family=bernoulli
link=logit
nrows=2000
ncols=3
formula=z+(1|fexplog(x,y))
colnames=x y z
covariance=-1 -1
mean=0 0
niter=50
maxiter=1
offset=0
predict=0
nrowspred=0
se=0
```
### X.csv
A csv file containing a matrix with the relevant covariates used to construct the fixed and random effects design matrices. Note that the columns of this matrix are identified by the colnames argument in the model.txt file.

### y.csv
A csv file containing the vector of outcome data.

### Other files
The program also accepts:
   - offset.csv   A vector containing the offset values. Set offset=1 in model.txt.
   - trials.csv    A vector containing the number of trials for each observation if using family=binomial
   - Xp.csv       A matrix of data for prediction. Set predict=0 to generate predictions.
   - offsetp.csv  If predicting and using and offset, the prediction value offsets should be specified in this file.

### Outputs
The program will write the results to a csv file (results.csv) in the same folder as the program which will be a table with the point estimates in the first column of fixed and covariance parameters, in that order and in the order they were specified, and the standard errors in the second column. The sampled random effect values are also written to u.csv. Predictions will also be included in this folder.

## License

[Your license here]

## Contact

[Your contact information]
```


