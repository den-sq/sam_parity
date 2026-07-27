#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARITY_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-}"
FRAMES_DIR="${FRAMES_DIR:-${SAM3_PROFILE_FRAMES_DIR:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SAM3_CHECKPOINT:-}}"
PROFILE_DIR="${PROFILE_DIR:-${PARITY_ROOT}/target/sam3-profiler}"
PROFILE_STEM="${PROFILE_STEM:-facebook_image_encoder_$(date -u +%Y%m%dT%H%M%SZ)}"
WARMUP="${WARMUP:-2}"
TARGET_FRAME="${TARGET_FRAME:-1}"

if [[ -z "${NSYS_BIN}" ]] || [[ ! -x "${NSYS_BIN}" ]]; then
  echo "Nsight Systems was not found; set NSYS_BIN to its executable" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Facebook profiler Python was not found; set PYTHON_BIN" >&2
  exit 1
fi
if [[ -n "${UPSTREAM_ROOT}" ]] && [[ ! -f "${UPSTREAM_ROOT}/sam3/__init__.py" ]]; then
  echo "Facebook SAM3 checkout was not found at ${UPSTREAM_ROOT}" >&2
  exit 1
fi
if [[ -z "${FRAMES_DIR}" ]]; then
  echo "Set FRAMES_DIR or SAM3_PROFILE_FRAMES_DIR to a prepared JPEG directory" >&2
  exit 1
fi
if [[ ! -f "${FRAMES_DIR}/000001.jpg" ]]; then
  echo "Profiler frames were not found at ${FRAMES_DIR}" >&2
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

if [[ -n "${UPSTREAM_ROOT}" ]]; then
  PROFILE_PYTHONPATH="${UPSTREAM_ROOT}:${PARITY_ROOT}/python"
else
  PROFILE_PYTHONPATH="${PARITY_ROOT}/python"
fi

PYTHONPATH="${PROFILE_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
"${NSYS_BIN}" profile \
  --trace=cuda-sw,cublas,cudnn,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --output="${PROFILE_DIR}/${PROFILE_STEM}" \
  "${PYTHON_BIN}" \
  -m sam3_parity.facebook_image_encoder_profile \
  --checkpoint "${CHECKPOINT_PATH}" \
  --frames-dir "${FRAMES_DIR}" \
  --target-frame "${TARGET_FRAME}" \
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
