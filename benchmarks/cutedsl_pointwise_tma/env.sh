# Source from an existing PyTorch environment; do not run the build setup script.
source "${POINTWISE_PERF_VENV:-/tmp/mjoux/venv}/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=/tmp/mjoux/.cache
export XDG_DATA_HOME=/tmp/mjoux/.local/share
export CUDA_CACHE_PATH=/tmp/mjoux/.nv
export PIP_CACHE_DIR=/tmp/mjoux/.pip
export TRITON_CACHE_DIR=/tmp/mjoux/.triton
export CCACHE_DIR=/tmp/mjoux/.ccache
export UV_CACHE_DIR=/tmp/mjoux/.uv
export TORCHINDUCTOR_CACHE_DIR=/tmp/mjoux/.inductor
export CUTE_DSL_CACHE_DIR=/tmp/mjoux/cutlass_python_cache
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Avoid Nsight's ldconfig child-process injection issue when the driver is here.
case "$(uname -m)" in
    aarch64) pointwise_driver_dir=/usr/lib/aarch64-linux-gnu ;;
    x86_64) pointwise_driver_dir=/usr/lib/x86_64-linux-gnu ;;
    *) pointwise_driver_dir= ;;
esac
if [[ -n "$pointwise_driver_dir" && -e "$pointwise_driver_dir/libcuda.so.1" ]]; then
    export TRITON_LIBCUDA_PATH="$pointwise_driver_dir"
fi
unset pointwise_driver_dir
