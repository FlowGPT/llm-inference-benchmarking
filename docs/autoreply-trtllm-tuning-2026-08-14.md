# AutoReply TensorRT-LLM 1.2.1 单卡调优记录

日期：2026-08-15
GPU：NVIDIA GeForce RTX 5090（单卡）
SLO：客户端 p50 E2E 严格小于 2.0 秒

## 结论

TensorRT-LLM 基线边界为 **5.0 PASS / 5.1 FAIL**。最佳启动配置的
三次独立冷启动确认均在 5.1 QPS 通过，5.2 QPS 未能重复通过，因此最终
边界是 **5.1 PASS / 5.2 FAIL**，相对基线提升 **+0.1 外部 QPS / +2.0%**。

本结果使用稳定镜像
`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`，镜像 digest 为
`sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9`。
模型、采样参数、`n=3` 和 `--max_seq_len 8192` 均未改变。

## 请求兼容适配

vLLM 用 `top_k=-1` 表示禁用 top-k，而 TensorRT-LLM 1.2.1 的 OpenAI
请求模型要求非负整数并以 `0` 为禁用默认值。实测结果如下：

| 线上的表示 | 结果 |
|---|---|
| `"top_k": -1` | HTTP 400，要求 `top_k >= 0` |
| `"top_k": null` | HTTP 400，整数 schema 校验失败 |
| 省略 `top_k` | HTTP 200，使用默认 `0`，即禁用 top-k |

因此控制器只在 TensorRT-LLM 路径把规范配置中的 Python `None` 在 JSON
序列化前省略。vLLM 和其他场景保持原逻辑。真实 replay、流式与非流式
请求均验证返回 `n=3` 的 choice indices `0,1,2`，其余采样字段不变。

## 测试方法

- 每个候选都删除并重建容器。
- 服务就绪后固定做 1 轮 3.0 QPS 预热；该轮不计入结果，用于消除首批
  CUDA Graph/shape 建立对连接数和延迟的干扰。
- 一级筛选使用 6 个 30 秒窗口、尾 3 轮 p50。
- 正式边界使用 12 个 30 秒窗口、尾 6 轮和自动稳态检测。
- 最终 PASS 点必须通过三次独立冷启动确认；相邻 `+0.1 QPS` 必须失败。
- QPS 是外部 HTTP 请求/秒；每请求固定 `n=3`，不能把约三倍的内部
  sequence rate 当作 QPS。

## 参数筛选过程

基线在 5.2 QPS 的筛选 p50 为 2.6286 秒。下表列出可比较的主要单项和
组合；“延迟改善”只说明 5.2 QPS 同点变化，不等于容量 QPS 收益。

| 候选 | 5.2/5.3 QPS 筛选 p50 (s) | 相对 5.2 基线 | 结论 |
|---|---:|---:|---|
| baseline @ 5.2 | 2.6286 | — | FAIL |
| CUDA Graph padding @ 5.2 | 2.5163 | 4.27% | 延迟改善，未过 SLO |
| dynamic max-token tuning @ 5.2 | 2.4674 | 6.13% | 延迟改善，未过 SLO |
| TRTLLMSampler @ 5.2 | 2.3190 | 11.78% | 延迟改善，未过 SLO |
| sampler + dynamic + graph padding @ 5.2 | 2.2156 | 15.72% | 延迟改善，未过 SLO |
| 上述组合 + `max_num_tokens=12288` @ 5.2 | 1.9779 | 跨过 SLO | PASS |
| 同配置 @ 5.3 | 3.1760 | — | FAIL |
| `max_num_tokens=16384` @ 5.3 | 2.5580 | — | FAIL，劣于 12288 |
| 再加 chunked prefill @ 5.3 | 1.9966 | — | 临界筛选 PASS |
| 再加 MAX_UTILIZATION @ 5.3 | 2.0025 | — | 临界 FAIL |
| 再加 async sampler worker @ 5.3 | 1.9985 | — | 临界筛选 PASS |
| 最终组合 @ 5.4 | 2.3851 | — | FAIL |

还覆盖了 KV cache fraction、dynamic moving-average window、stream interval、
postprocess worker、attention backend 等参数。KV fraction 0.95 没有提高边界；
postprocess worker 产生无内容 chunk，判为不兼容；FLASHINFER attention
启动日志显示 `cache_reuse=False`，与生产前缀复用语义不一致，未进入决赛。
早期一次未预热的 stream-interval 结果连接数异常，只保留为非可比诊断，
不用于排名。

## 正式边界

| 配置 | QPS | 状态 | 主判据 p50 (s) | 尾 6 轮 p50 (s) |
|---|---:|---:|---:|---:|
| baseline | 5.0 | PASS | 1.5039 | 1.5747 |
| baseline | 5.1 | FAIL | 2.0490 | 2.0490 |
| winner formal | 5.3 | FAIL | 2.5902 | 2.5902 |
| winner 5.2 confirmation 1 | 5.2 | PASS | 1.7875 | 2.0749 |
| winner 5.2 confirmation 2 | 5.2 | FAIL | 2.1664 | 2.1664 |
| winner confirmation 1 | 5.1 | PASS | 1.7591 | 1.8381 |
| winner confirmation 2 | 5.1 | PASS | 1.7580 | 1.8414 |
| winner confirmation 3 | 5.1 | PASS | 1.8570 | 1.8570 |

5.2 只通过一次、第二次失败，所以不能宣称为稳定容量。5.1 三次全部通过，
且基线 5.1 失败，构成可复现的 **+0.1 QPS / +2.0%** 提升。

## 前缀缓存对齐

TensorRT-LLM 1.2.1 的 `/metrics` 在该服务模式返回空 body，无法像 vLLM
一样读取 token 级命中计数，因此没有伪造 66–67% 数值。最佳配置启动日志
明确为 `cache_reuse=True`，真实请求也保持 `n=3`；会关闭复用的 FLASHINFER
候选已排除。vLLM 的约 66.45% 命中率仍由同请求三个分支中后两个复用
prompt 解释，而非多轮历史复用。

## 最佳配置

`/run/autoreply-options.yml`：

```yaml
sampler_type: TRTLLMSampler
sampler_force_async_worker: true
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  dynamic_batch_config:
    enable_batch_size_tuning: true
    enable_max_num_tokens_tuning: true
    dynamic_batch_moving_average_window: 128
cuda_graph_config:
  enable_padding: true
```

启动命令：

```bash
docker run -d --init --name autoreply-m12-trtllm-winner \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONPATH=/opt/autoreply -p 8080:8000 \
  -v /root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8:/models/kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4:ro \
  -v /root/llm-inference-benchmarking/scripts/trtllm_autoreply_tokenizer.py:/opt/autoreply/trtllm_autoreply_tokenizer.py:ro \
  -v /root/llm-inference-benchmarking/bench-runs/autoreply-framework-shootout-20260814/trtllm/candidates/winner-async-sampler/options.yml:/run/autoreply-options.yml:ro \
  -w /models nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
  trtllm-serve serve kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4 \
  --backend pytorch \
  --custom_module_dirs /opt/autoreply \
  --custom_tokenizer trtllm_autoreply_tokenizer.AutoReplyTokenizer \
  --host 0.0.0.0 --port 8000 \
  --max_batch_size 96 --max_num_tokens 12288 --max_seq_len 8192 \
  --kv_cache_free_gpu_memory_fraction 0.90 \
  --config /run/autoreply-options.yml --enable_chunked_prefill
```

## 与 vLLM 决赛

vLLM 最佳边界为 9.4 QPS，TensorRT-LLM 为 5.1 QPS。vLLM 多
**4.3 外部 QPS**，相对 TensorRT-LLM 高 **84.31%**；反向看，TensorRT-LLM
比 vLLM 低 **45.74%**。因此该模型、该请求分布和 RTX 5090 上的最终框架
选择仍是 vLLM 0.27.1。
