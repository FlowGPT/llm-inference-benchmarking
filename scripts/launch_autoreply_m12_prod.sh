#!/usr/bin/env bash
# Launch AutoReply M12 NVFP4 to match production vLLM flags.
# Does not stop other containers. Refuses to start if GPU 0 is already occupied.
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
NAME="${CONTAINER_NAME:-autoreply-m12-prod}"
HOST_PORT="${HOST_PORT:-8080}"
CONTAINER_PORT=8080
HF_CACHE="${HOME}/.cache/huggingface"
MODEL_PATH="/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8"
SERVED_NAME="kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4"
API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY before launching the service}"

if [[ ! -f "${HF_CACHE}/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8/model.safetensors" ]]; then
  echo "missing local model at ${HF_CACHE}/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already running" >&2
  docker ps --filter "name=^${NAME}$"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "removing leftover stopped container $NAME"
  docker rm "$NAME" >/dev/null
fi

gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${gpu_used:-0}" -gt 2048 ]]; then
  echo "GPU 0 is occupied (${gpu_used} MiB used). Running containers:" >&2
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  echo "Stop the occupant first, then rerun. This script will not kill other containers." >&2
  exit 2
fi

exec docker run -d --rm \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$CONTAINER_PORT" \
  --dtype auto \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 96 \
  --max-num-batched-tokens 8192 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.94 \
  --async-scheduling \
  --api-key "$API_KEY" \
  --stream-interval 5 \
  -O3
