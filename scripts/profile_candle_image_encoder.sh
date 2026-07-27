#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARITY_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

CUDA_ROOT="${CUDA_ROOT:-${CUDA_HOME:-/usr/local/cuda}}"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
FRAMES_DIR="${FRAMES_DIR:-${SAM3_PROFILE_FRAMES_DIR:-}}"
FRAME_NAME="${FRAME_NAME:-000001.jpg}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SAM3_CHECKPOINT:-}}"
PROFILE_DIR="${PROFILE_DIR:-${PARITY_ROOT}/target/sam3-profiler}"
PROFILE_STEM="${PROFILE_STEM:-candle_image_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP="${WARMUP:-2}"

if [[ -z "${NSYS_BIN}" ]] || [[ ! -x "${NSYS_BIN}" ]]; then
  echo "Nsight Systems was not found; set NSYS_BIN to its executable" >&2
  exit 1
fi
if [[ -z "${FRAMES_DIR}" ]]; then
  echo "Set FRAMES_DIR or SAM3_PROFILE_FRAMES_DIR to a prepared JPEG directory" >&2
  exit 1
fi
if [[ ! -f "${FRAMES_DIR}/${FRAME_NAME}" ]]; then
  echo "Profiler frame was not found at ${FRAMES_DIR}/${FRAME_NAME}" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH or SAM3_CHECKPOINT to a SAM3 checkpoint" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Profiler checkpoint was not found at ${CHECKPOINT_PATH}" >&2
  exit 1
fi

NSYS_CONFIG_FILE="${NSYS_CONFIG_FILE:-$("${NSYS_BIN}" -z)}"
mkdir -p "${PROFILE_DIR}"
mkdir -p "$(dirname -- "${NSYS_CONFIG_FILE}")"
if [[ ! -f "${NSYS_CONFIG_FILE}" ]] ||
  ! grep -qxF 'CuptiUseRawGpuTimestamps=false' "${NSYS_CONFIG_FILE}"; then
  printf '%s\n' 'CuptiUseRawGpuTimestamps=false' >>"${NSYS_CONFIG_FILE}"
fi

PATH="${CUDA_ROOT}/bin:${PATH}" \
CUDA_HOME="${CUDA_ROOT}" \
CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-75}" \
cargo build \
  --manifest-path "${PARITY_ROOT}/Cargo.toml" \
  --release \
  --features cuda \
  --bin sam3_image_encoder_profile

"${NSYS_BIN}" profile \
  --trace=cuda-sw,cublas,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --output="${PROFILE_DIR}/${PROFILE_STEM}" \
  "${PARITY_ROOT}/target/release/sam3_image_encoder_profile" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --frame "${FRAMES_DIR}/${FRAME_NAME}" \
  --dtype f32 \
  --warmup "${WARMUP}"

REPORT="${PROFILE_DIR}/${PROFILE_STEM}.nsys-rep"
SUMMARY="${PROFILE_DIR}/${PROFILE_STEM}.stats.txt"
"${NSYS_BIN}" stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  "${REPORT}" | tee "${SUMMARY}"
if ! grep -q 'CUDA GPU Kernel Summary' "${SUMMARY}"; then
  echo "Nsight report did not contain CUDA kernel timing data" >&2
  exit 1
fi

echo "Nsight report: ${REPORT}"
echo "Text summary: ${SUMMARY}"
