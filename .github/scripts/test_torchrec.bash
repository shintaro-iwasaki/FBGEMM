#!/bin/bash

# Exit on failure
set -e

. "$(dirname "$(realpath -s "$0")")/setup_env.bash"


verbose=0
python_version=""
pytorch_cuda_version="x"
fbgemm_wheel_path="x"
miniconda_prefix="${HOME}/miniconda"

usage () {
  echo "Usage: bash test_wheel.bash -p PYTHON_VERSION -P PYTORCH_CHANNEL_NAME -c PYTORCH_CUDA_VERSION -w FBGEMM_WHEEL_PATH [-m MINICONDA_PREFIX] [-v] [-h]"
  echo "-v                  : verbose"
  echo "-h                  : help"
  echo "PYTHON_VERSION      : Python version (e.g., 3.7, 3.8, 3.10)"
  echo "PYTORCH_CHANNEL_NAME: PyTorch's channel name (e.g., pytorch-nightly, pytorch-test (=pre-release), pytorch (=stable release))"
  echo "PYTORCH_CUDA_VERSION: PyTorch's CUDA version (e.g., 11.6, 11.7)"
  echo "FBGEMM_WHEEL_PATH   : path to FBGEMM_GPU's wheel file"
  echo "MINICONDA_PREFIX    : path to install Miniconda (default: \$HOME/miniconda)"
  echo "Example 1: Python 3.10 + CUDA 11.7, install miniconda at /home/user/tmp/minoconda, using dist/fbgemm_gpu.whl"
  echo "       bash test_wheel.bash -v -p 3.10 -c 11.7 -m /home/user/tmp/minoconda -w dist/fbgemm_gpu.whl"
  echo "Example 2: Python 3.10 + CPU, install miniconda at \$HOME/miniconda, using /tmp/fbgemm_gpu_cpu.whl"
  echo "       bash test_wheel.bash -v -p 3.10 -c \"\" -w /tmp/fbgemm_gpu_cpu.whl"
}

while getopts vhp:P:c:m:w: flag
do
    case "$flag" in
        v) verbose="1";;
        p) python_version="${OPTARG}";;
        P) pytorch_channel_name="${OPTARG}";;
        c) pytorch_cuda_version="${OPTARG}";;
        m) miniconda_prefix="${OPTARG}";;
        w) fbgemm_wheel_path="${OPTARG}";;
        h) usage
           exit 0;;
        *) usage
           exit 1;;
    esac
done

if [ x"$python_version" == x -o x"$pytorch_cuda_version" == x"x" -o x"$miniconda_prefix" == x -o x"$pytorch_channel_name" == x -o x"$fbgemm_wheel_path" == x ]; then
  usage
  exit 1
fi
python_tag=$(echo ${python_version} | sed "s/\.//")

if [ x"$verbose" == x1 ]; then
  # Print each line verbosely
  set -x -e
fi

################################################################################
echo "## 0. Minimal check"
################################################################################

if [ ! -d "torchrec" ]; then
  echo "Error: this script must be executed in torchrec/"
  exit 1
fi

################################################################################
echo "## 1. Set up Miniconda"
################################################################################

setup_miniconda "$miniconda_prefix"

################################################################################
echo "## 2. Create test_binary environment"
################################################################################

create_conda_environment test_binary "$python_version" "$pytorch_channel_name" "$pytorch_cuda_version"

# Comment out FBGEMM_GPU since we will install it from "$fbgemm_wheel_path"
sed -i 's/fbgemm_gpu/#fbgemm_gpu/g' requirements.txt
conda run -n test_binary python -m pip install -r requirements.txt
# Install FBGEMM_GPU from a local wheel file.
conda run -n test_binary python -m pip install "$fbgemm_wheel_path"
conda run -n test_binary python -c "import fbgemm_gpu"

################################################################################
echo "## 3. Build TorchRec"
################################################################################

rm -rf dist
conda run -n test_binary python setup.py bdist_wheel --package_name torchrec --python-tag="py${python_tag}"

################################################################################
echo "## 4. Import TorchRec"
################################################################################

conda run -n test_binary python -m pip install dist/torchrec*.whl
conda run -n test_binary python -c "import torchrec"

echo "Test succeeded"
