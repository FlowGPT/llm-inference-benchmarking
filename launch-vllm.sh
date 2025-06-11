export HF_HOME=/home/paperspace/qq/hf_cache/

python3 -m vllm.entrypoints.openai.api_server \
 --model wangqia0309/Captain-Eris_Violet-V0.420-12B-FP8-Dynamic \
 --host 0.0.0.0 \
 --port 8000 \
 --max-model-len 9000 \
 --swap-space 4 \
 --enable-chunked-prefill \
 --disable-log-requests \
 --enable-prefix-caching \
 --max-num-seqs 72 \
 --quantization compressed-tensors \
 --max-num-batched-tokens 1024 \
 --kv-cache-dtype fp8
