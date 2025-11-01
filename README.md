# glmmrGPU - GPU-Accelerated Mixed Model Fitting

GPU-accelerated mixed model fitting using CUDA, cuSolver, and Eigen.

## Prerequisites - Windows

Before building, ensure you have the following installed:

### Required Software

1. **Visual Studio 2019 or 2022** (Community Edition is free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select "Desktop development with C++"
   - Make sure to include the latest Windows SDK

2. **CUDA Toolkit 11.0 or later**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Recommended: CUDA 12.x
   - Verify installation:
```powershell
     nvcc --version
```

3. **CMake 3.18 or later**
   - Download: https://cmake.org/download/
   - During installation, select "Add CMake to system PATH"
   - Verify installation:
```powershell
     cmake --version
```

4. **Git** (optional, for cloning)
   - Download: https://git-scm.com/downloads
   - Or download project as ZIP from repository

### Hardware Requirements

- NVIDIA GPU with compute capability 5.2 or higher
- Verify your GPU:
```powershell
  nvidia-smi
```

## Quick Start

### 1. Get the Code

**Option A: Clone with Git**
```powershell
git clone https://github.com/yourusername/glmmrGPU.git
cd glmmrGPU
```

**Option B: Download ZIP**
- Download ZIP from repository
- Extract to a folder (e.g., `C:\Projects\glmmrGPU`)
- Open PowerShell in that folder

### 2. Build the Project
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

### 3. Run the Program
```powershell
# From the build directory
.\Release\glmmrGPU.exe ..\data

# Or with absolute path
.\Release\glmmrGPU.exe C:\Projects\glmmrGPU\data
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
./glmmrGPU ../data
./glmmrGPU /home/user/data
```

## Project Structure
```
glmmrGPU/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/
│   └── cuda_cholesky.h     # Public API
├── src/
│   ├── cuda_cholesky.cu    # CUDA implementation
│   └── main.cpp            # Main program
├── data/                   # Input data files
│   ├── matrix_A.txt
│   ├── matrix_B.txt
│   └── vector_b.txt
└── build/                  # Build output (generated)
    ├── Debug/
    │   └── glmmrGPU.exe
    └── Release/
        └── glmmrGPU.exe
```

## Advanced Options

### Build with Specific CUDA Architecture

If you want to target specific GPU architecture:
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES="86"
```

Common architectures:
- `52` - GTX 900 series (Maxwell)
- `61` - GTX 10 series (Pascal)
- `75` - RTX 20 series (Turing)
- `86` - RTX 30 series (Ampere)
- `89` - RTX 40 series (Ada Lovelace)

### Build with Custom Boost/Eigen

If you have Boost or Eigen installed elsewhere:
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DBOOST_ROOT="C:/Libraries/boost_1_83_0" ^
    -DEIGEN3_INCLUDE_DIR="C:/Libraries/eigen"
```

### Verbose Build Output

To see detailed compilation commands:
```powershell
cmake --build . --config Release --verbose
```

## Building in Visual Studio (GUI)

If you prefer using Visual Studio IDE:

1. **Generate solution:**
```powershell
   cmake -B build -G "Visual Studio 17 2022" -A x64
```

2. **Open solution:**
```powershell
   start build\glmmrGPU.sln
```

3. **In Visual Studio:**
   - Select configuration (Debug/Release) from toolbar
   - Press `Ctrl+Shift+B` to build
   - Press `F5` to run with debugging

## System Requirements

### Minimum
- Windows 10 (64-bit)
- NVIDIA GPU with compute capability 5.2+
- 16 GB RAM
- 64 GB free disk space

### Recommended
- Windows 10/11 (64-bit)
- NVIDIA RTX series GPU
- 64 GB RAM
- SSD with 5 GB free space

## License

[Your license here]

## Contact

[Your contact information]
```


