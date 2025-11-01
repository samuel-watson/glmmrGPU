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

## Building Step-by-Step

### First-Time Setup

1. **Open PowerShell** (or Command Prompt)

2. **Navigate to project directory:**
```powershell
   cd C:\Projects\glmmrGPU
```

3. **Create and enter build directory:**
```powershell
   mkdir build
   cd build
```

4. **Configure project with CMake:**
```powershell
   cmake .. -G "Visual Studio 17 2022" -A x64
```
   
   This will:
   - Detect your CUDA installation
   - Download Eigen and Boost automatically (first time only)
   - Generate Visual Studio solution files

5. **Build the project:**
```powershell
   # Build Release version (optimized, faster)
   cmake --build . --config Release
   
   # Or build Debug version (with debug symbols)
   cmake --build . --config Debug
```

6. **The executable will be at:**
   - Release: `build\Release\glmmrGPU.exe`
   - Debug: `build\Debug\glmmrGPU.exe`

### Rebuilding After Changes

If you modify source files:
```powershell
# From build directory
cmake --build . --config Release
```

### Clean Rebuild

If you encounter issues, do a clean rebuild:
```powershell
# Remove build directory
cd ..
Remove-Item -Recurse -Force build

# Rebuild from scratch
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Usage

### Basic Usage
```powershell
glmmrGPU.exe <data_directory>
```

**Example:**
```powershell
# Using relative path
.\Release\glmmrGPU.exe ..\data

# Using absolute path
.\Release\glmmrGPU.exe C:\Projects\glmmrGPU\data
```

### Input File Format

The program expects the following files in the data directory:

**matrix_A.txt:**
```
4 4
4 2 1 1
2 5 3 2
1 3 6 4
1 2 4 7
```

Format: First line is `rows columns`, followed by matrix data.

### Output

Results are saved in the data directory as `output_*.txt`.

## Troubleshooting

### Issue: "nvcc not found"

**Solution:** CUDA not in PATH. Add manually:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
```

Or permanently:
- System Properties → Environment Variables
- Edit PATH → Add CUDA bin directory

### Issue: "CMake Error: Could not find CMAKE_CUDA_COMPILER"

**Solution:** Install CUDA Toolkit or specify CUDA location:
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe"
```

### Issue: "MSBuild not found"

**Solution:** Visual Studio not installed or not in PATH. Either:
- Reinstall Visual Studio with C++ tools
- Or use Visual Studio Developer Command Prompt

To open Developer Command Prompt:
- Start Menu → Visual Studio 2022 → Developer Command Prompt
- Then navigate to project and build

### Issue: "error C1060: compiler is out of heap space"

**Solution:** Already handled by `/bigobj` flag in CMakeLists.txt. If still occurs:
```powershell
# Clean rebuild
Remove-Item -Recurse -Force build
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Issue: "Cannot open data file"

**Solution:** Specify correct path to data directory:
```powershell
# Check current directory
Get-Location

# Use absolute path
.\Release\glmmrGPU.exe C:\Projects\glmmrGPU\data
```

### Issue: Dependencies fail to download (Eigen/Boost)

**Symptom:** CMake hangs or fails downloading Eigen/Boost

**Solution:** Check internet connection or download manually:

1. Download Eigen: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
2. Download Boost: https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz
3. Extract to:
   - `build\_deps\eigen\`
   - `build\_deps\boost\`
4. Rebuild

### Issue: "This application requires the CUDA Runtime"

**Solution:** CUDA Runtime not installed. Install CUDA Toolkit from NVIDIA.

### Issue: Build works but crashes on run

**Possible causes:**
1. **No NVIDIA GPU:** Program requires NVIDIA GPU
2. **Old GPU drivers:** Update to latest NVIDIA drivers
3. **Data files missing:** Ensure data directory contains required files

**Check GPU:**
```powershell
nvidia-smi
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
- 4 GB RAM
- 2 GB free disk space

### Recommended
- Windows 10/11 (64-bit)
- NVIDIA RTX series GPU
- 8 GB RAM
- SSD with 5 GB free space

## Performance Tips

1. **Use Release build for production:**
   - Release builds are significantly faster (10-100x)
   - Debug builds include extra checks and symbols

2. **Close other GPU applications:**
   - Close games, video editors, or other GPU-intensive apps
   - Check with `nvidia-smi`

3. **Update GPU drivers:**
   - Latest drivers can improve performance
   - Download from: https://www.nvidia.com/drivers

## Getting Help

If you encounter issues:

1. **Check this README** for common solutions
2. **Look at error messages** carefully
3. **Verify prerequisites** are installed correctly
4. **Try clean rebuild** (delete build folder)
5. **Check GitHub Issues** for similar problems

## License

[Your license here]

## Contact

[Your contact information]
```

## Quick Reference Card

Create a separate **QUICKSTART.txt**:
```
========================================
glmmrGPU - Quick Start Guide (Windows)
========================================

INSTALL PREREQUISITES:
1. Visual Studio 2022 (with C++)
2. CUDA Toolkit 12.x
3. CMake 3.18+

BUILD STEPS:
1. cd C:\path\to\glmmrGPU
2. mkdir build
3. cd build
4. cmake .. -G "Visual Studio 17 2022" -A x64
5. cmake --build . --config Release

RUN:
.\Release\glmmrGPU.exe ..\data

REBUILD (after code changes):
cmake --build . --config Release

CLEAN REBUILD (if problems):
cd ..
Remove-Item -Recurse build
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

COMMON ISSUES:
- "nvcc not found" → Install CUDA, add to PATH
- Build hangs → Check internet (downloading dependencies)
- Crashes → Check nvidia-smi, update drivers
- Data not found → Use absolute path to data dir

HELP: See README.md for detailed instructions
========================================
