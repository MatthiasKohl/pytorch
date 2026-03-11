# Static analysis of element-wise lambda functions

First set up a dev environment, for this see [Install from source](../../README.md#from-source),
and specifically [Install Dependencies](../../README.md#install-dependencies)
for the current up-to-date version

To build, use the following config:

```bash

# set up dev environment (see above)
export PYTORCH_REPO=/path/to/pytorch
export BUILD_DIR="${PYTORCH_REPO}/build"
cd "${PYTORCH_REPO}"
source /path/to/venv/bin/activate
pip3 install --upgrade pip
pip3 install uv
uv pip install --group dev

# use 9.0 for Hopper etc.
export TORCH_CUDA_ARCH_LIST="10.0"
export CUDA_HOME=/usr/local/cuda
cmake -B "${BUILD_DIR}" \
      -S "${PYTORCH_REPO}" \
      -G Ninja \
      -DUSE_CUDA=ON \
      -DBUILD_SIMPLE_GPU_KERNELS=ON \
      -DBUILD_PYTHON=OFF \
      -DBUILD_TEST=OFF \
      -DUSE_DISTRIBUTED=OFF \
      -DUSE_MKLDNN=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_XNNPACK=OFF \
      -DUSE_FBGEMM=OFF \
      -DUSE_KINETO=OFF \
      -DUSE_CUPTI_SO=OFF \
      -DUSE_GFLAGS=OFF \
      -DUSE_GLOG=OFF \
      -DUSE_SYSTEM_NCCL=OFF \
      -DUSE_MAGMA=OFF \
      -DUSE_CUSPARSELT=OFF \
      -DUSE_NATIVE_ARCH=OFF \
      -DUSE_VALGRIND=OFF \
      -DUSE_ITT=OFF \
      -DUSE_C10D_UCC=OFF \
      -DUSE_GLOO=OFF \
      -DUSE_C10D_MPI=OFF \
      -DONNX_ML=OFF \
      -DUSE_TENSORPIPE=OFF \
      -DBUILD_NVFUSER=OFF \
      -DBUILD_FUNCTORCH=ON \
      -DBUILD_JNI=OFF \
      -DUSE_MPS=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_OPENMP=OFF

cmake --build "${BUILD_DIR}" --target ATEN_CPU_FILES_GEN_TARGET

cmake --build "${BUILD_DIR}" --target simple_gpu_kernels -- -j$(nproc)

# repeat above for various architectures and move libraries, e.g.
mv "${BUILD_DIR}/gpu_kernel_lib/libsimple_gpu_kernels.a" ./libsimple_gpu_kernels_sm100.a

# finally call the parse script on all architectures
python tools/parse_sass.py libsimple_gpu_kernels_sm100.a -o sass_summary_sm100.txt --csv sass_summary.csv
# repeat above for all archs keeping same csv output file
```
