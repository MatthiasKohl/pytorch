# Use an activated environment, or provide its location explicitly.
if [[ -n "${POINTWISE_PERF_VENV:-}" ]]; then
    source "$POINTWISE_PERF_VENV/bin/activate"
fi
: "${POINTWISE_PERF_CACHE_ROOT:?Set this to an allowed temporary cache directory}"
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$POINTWISE_PERF_CACHE_ROOT/.cache"
export XDG_DATA_HOME="$POINTWISE_PERF_CACHE_ROOT/.local/share"
export CUDA_CACHE_PATH="$POINTWISE_PERF_CACHE_ROOT/.nv"
export PIP_CACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/.pip"
export TRITON_CACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/.triton"
export CCACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/.ccache"
export UV_CACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/.uv"
export TORCHINDUCTOR_CACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/.inductor"
export CUTE_DSL_CACHE_DIR="$POINTWISE_PERF_CACHE_ROOT/cutlass_python_cache"
export POINTWISE_PERF_OUTPUT_DIR="${POINTWISE_PERF_OUTPUT_DIR:-$POINTWISE_PERF_CACHE_ROOT/pointwise-results}"
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
