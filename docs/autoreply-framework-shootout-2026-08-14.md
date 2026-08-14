# AutoReply 单卡推理框架调优与决赛

日期：2026-08-14
GPU：NVIDIA GeForce RTX 5090（单卡）
SLO：客户端 p50 E2E 严格小于 2.0 秒

## 不可变实验契约

- 模型检查点：`/root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8`
- 服务模型名：`kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4`
- 数据集：`datasets/autoreply_prod_dist_repeated_13x.jsonl`，由固定的 1,000 条生产分布样本重复 13 轮，共 13,000 行
- 对外发布数据集：`/mnt/shared/sss/data/auto-reply-test.json`
- 请求参数：`n=3`、`max_tokens=50`、`temperature=0.7`、`top_p=0.8`、`frequency_penalty=0.01`、`presence_penalty=0.01`、禁用 `min_p`、`top_k=-1`、`stop=["<|im_end|>"]`
- 模型最大长度固定为 8,192：vLLM 使用 `--max-model-len 8192`，TensorRT-LLM 使用 `--max_seq_len 8192`；该参数不进入搜索空间
- QPS 统一指客户端发出的 HTTP 请求/秒。因为每个请求固定 `n=3`，服务端序列速率约为外部 QPS 的三倍，不能作为 QPS 结果
- 每个测试点销毁并重新创建任务容器；任何时刻只运行一个框架实例

## 为什么前缀缓存命中稳定在 66–67%

这个命中率主要不是多轮历史或跨请求复用造成的，而是固定 `n=3` 的同请求内分支复用。一个外部请求产生三条具有完全相同 prompt 的生成序列：第一条建立 KV 前缀，后两条复用，因此理想命中率为：

```text
(3 - 1) / 3 = 2 / 3 = 66.667%
```

基线筛选原始计数为 11,112,608 个命中 token / 16,723,773 个查询 token，即 66.44797%。实际值略低于 2/3，是因为 vLLM 只缓存完整 KV block，prompt 末尾不足一块的 token 不能计为命中，再叠加极少量冷启动边界。各正常候选的结果都稳定在约 66.45%，证明请求形状与缓存语义一致。`long-prefill1024/2048` 的命中率异常下降则伴随严重排队，因而被判为无效候选。

## vLLM 0.27.1 参数审计与一级筛选

从镜像的完整 CLI 帮助中提取并处置了 370 个长参数：74 个归为适用搜索参数、27 个诊断/控制参数、5 个固定不变量、264 个与单卡稠密纯文本场景无关。完整清单位于运行产物的 `vllm/manifest/flag_compatibility.{json,md}`。

一级筛选使用 9.4 外部 QPS、6 轮、尾部 3 轮 p50 判定；它只用于晋级，不直接当作最大容量结论。

| 候选 | 参数族 | 状态 | p50 E2E (s) | 前缀命中率 |
|---|---|---:|---:|---:|
| scheduled4096 | scheduled tokens | PASS | 1.7518 | 66.45% |
| scheduled4096 + api2 + throughput | interaction | PASS | 1.7532 | 66.45% |
| scheduled4096 + throughput | interaction | PASS | 1.7814 | 66.45% |
| scheduled4096 + api2 | interaction | PASS | 1.8177 | 66.45% |
| api2 | frontend | PASS | 1.8687 | 66.45% |
| performance-throughput | graph/compile | PASS | 1.8698 | 66.45% |
| linear-cutlass | kernel | PASS | 1.8939 | 66.45% |
| baseline | baseline | PASS | 1.9137 | 66.45% |
| cumem-on | allocator | PASS | 1.9403 | 66.45% |
| scheduled6144 | scheduled tokens | PASS | 1.9492 | 66.45% |
| seq128 | sequence capacity | PASS | 1.9505 | 66.45% |
| reserve-full-isl-off | prefill | PASS | 1.9768 | 66.45% |
| stream20 | frontend | PASS | 1.9997 | 66.45% |
| stream10 | frontend | FAIL | 2.0005 | 66.45% |
| renderer4 | frontend | FAIL | 2.0094 | 66.45% |
| gpu95 | memory | FAIL | 2.0135 | 66.45% |
| block16 | KV layout | FAIL | 2.0365 | 66.45% |
| chunked-off | prefill | FAIL | 2.0645 | 66.45% |
| renderer8 | frontend | FAIL | 2.0826 | 66.45% |
| seq160 | sequence capacity | FAIL | 2.0917 | 66.45% |
| seq64 | sequence capacity | FAIL | 2.2241 | 66.45% |
| block32 | KV layout | FAIL | 2.4391 | 66.16% |
| batch12288 | batch tokens | FAIL | 2.4622 | 66.45% |
| gpu96 | memory | FAIL | 2.6106 | 66.45% |
| async-off | scheduler | FAIL | 3.1899 | 66.45% |
| eager-control | graph/compile | FAIL | 5.0315 | 66.45% |
| long-prefill2048 | prefill | FAIL | 45.6215 | 55.60% |
| long-prefill1024 | prefill | FAIL | 133.4204 | 33.78% |
| batch16384 | batch tokens | ERROR | — | — |
| linear-fi-b12x | kernel | ERROR | — | — |
| linear-fi-cutedsl | kernel | ERROR | — | — |
| linear-fi-trtllm | kernel | ERROR | — | — |
| ubatch8 / ubatch16 | microbatch | ERROR | — | — |

主要错误不是静默回退：`batch16384` 在 FP4 内核预热阶段 OOM；显式 FlashInfer NVFP4 内核与 RTX 5090 的 SM 12.x 能力组合不兼容；`ubatch` 仅支持特定 expert-parallel all-to-all 后端，不适用于本单卡稠密模型。自动选择与 CUTLASS 路径可正常运行。

筛选显示最强单项是 `--max-num-scheduled-tokens 4096`。关闭异步调度或 CUDA Graph 会大幅退化；提高内存占比、盲目扩大批 token、改变 KV block、关闭 chunked prefill 都没有收益。三个有收益旋钮的交互组合没有显著优于 scheduler 单项，因此最终仍需通过最大可持续 QPS 二分判断是否存在饱和区间收益。

## 最大可持续 QPS 二分

每个点使用 12 个连续 30 秒到达窗口，只用尾部 6 轮 p50；每个点均冷重启服务。初始低/高点为 8.5/10.5 QPS，精度 0.1 QPS。最终结果还需在最佳通过点和相邻 `+0.1 QPS` 失败点复验。

基线已确认：

- 8.5 QPS：PASS，尾部 p50 0.8876 秒，前缀命中 66.4477%
- 10.5 QPS：FAIL，尾部 p50 27.4992 秒，前缀命中 66.4474%；持续到达期间等待队列累积，属于容量过载

完整二分结果如下。表中的“收益”只按最大合规外部 QPS 计算；同档位内的延迟改善不计作 QPS 收益。

| 候选 | 最佳 PASS | 相邻 FAIL | PASS 点 p50 (s) | 相对基线收益 |
|---|---:|---:|---:|---:|
| baseline | 9.3 | 9.4 | 1.5897 | — |
| api2 | 9.3 | 9.4 | 1.5973 | 0.0 QPS / 0.00% |
| scheduled4096 | 9.4 | 9.5 | 1.8425 | **+0.1 QPS / +1.08%** |
| performance-throughput | 9.4 | 9.5 | 1.9583 | **+0.1 QPS / +1.08%** |
| linear-cutlass | 9.4 | 9.5 | 1.9877 | **+0.1 QPS / +1.08%** |
| scheduled4096 + api2 + throughput | 9.4 | 9.5 | 1.8470 | **+0.1 QPS / +1.08%** |

四个配置都把容量边界提高了 0.1 QPS，但组合没有继续跨过 9.5 QPS，不能相加为 0.4 QPS。最终选择最简单、在 9.4 QPS 留有最大延迟余量的 `scheduled4096`。

## 最终独立冷启动确认

每次确认都删除并重建容器，保持 12 轮、尾部 6 轮判定。

| 测试 | QPS | 结果 | p50 E2E (s) | 前缀命中率 |
|---|---:|---:|---:|---:|
| PASS 复验 1 | 9.4 | PASS | 1.8084 | 66.4476% |
| PASS 复验 2 | 9.4 | PASS | 1.8789 | 66.4476% |
| PASS 复验 3 | 9.4 | PASS | 1.8713 | 66.4476% |
| 相邻失败点 | 9.5 | FAIL | 2.1383 | 66.4476% |

因此 vLLM 0.27.1 的最终严格边界为 **9.4 PASS / 9.5 FAIL**。对比基线 9.3 PASS / 9.4 FAIL，最佳配置提升 **0.1 外部 QPS（1.08%）**。

## TensorRT-LLM 兼容性门禁

截至实验日，官方最新稳定镜像为 `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`；同时用 `1.3.0rc22` 做了预发布回退验证。版本依据来自 [TensorRT-LLM Releases](https://github.com/NVIDIA/TensorRT-LLM/releases) 与 [NGC 镜像标签](https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release/-/tags?_lr=1)。

- 1.2.1 成功识别 Mistral、NVFP4 权重与 FP8 KV；内置 Transformers 4.57.3 不认识检查点的 `TokenizersBackend`。兼容适配器直接读取同一 `tokenizer.json`，验证 token ID 与 chat template 完全一致后，服务成功启动。
- 1.3.0rc22 的 Transformers 5.5.4 可原生加载 tokenizer，模型也成功启动。
- 两个版本对固定请求都返回 HTTP 400：`require top_k >= 0, got top_k=-1`。
- TensorRT-LLM 的禁用哨兵是 `top_k=0`，但把固定请求的 `-1` 改为 `0` 会违反实验契约，而且不是启动参数调优。因此兼容性门禁失败，未获准进入 QPS 搜索，也不虚构 TensorRT-LLM 的对比 QPS。
- CLI 按约束传入 `--max_seq_len 8192`。稳定版缓存管理器内部日志显示 8193，而 LLM Args 与 CUDA Graph 仍显示 8192；这是内部额外槽位表现，不是本实验调大参数。

## 决赛结论

在“固定请求必须原样被接受”的前提下，唯一可部署并完成严格容量验证的框架是 **vLLM 0.27.1**。TensorRT-LLM 不是因为实测 QPS 更低而落败，而是稳定版与 RC 都未通过固定 API 请求兼容性门禁；因此本报告不提供不公平的跨框架吞吐数字。

最佳启动命令：

```bash
docker run -d --init --name autoreply-m12-vllm-scheduled4096 \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8080:8080 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python3 vllm/vllm-openai:v0.27.1 \
  -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/saved_models_user-model-afs-sfw-m12-v7_1s1600_nvfp4_kv_fp8 \
  --served-model-name kaonai/user-model-afs-sfw-m12-v7.1s1600-nvfp4 \
  --host 0.0.0.0 --port 8080 --dtype auto \
  --max-model-len 8192 \
  --enable-chunked-prefill --enable-prefix-caching \
  --max-num-seqs 96 --max-num-batched-tokens 8192 \
  --quantization modelopt --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.94 --async-scheduling \
  --api-key '<set-locally>' --stream-interval 5 -O3 \
  --max-num-scheduled-tokens 4096
```

## 投机采样调研（未训练、未计收益）

首选 NVIDIA Model Optimizer EAGLE3：官方训练支持矩阵明确包含 Mistral，并提供在线、离线、流式 hidden-state、草稿词表压缩以及 TensorRT-LLM/SGLang 部署路径。vLLM Speculators 是 vLLM 原生备选，但其当前支持表仍将 Mistral 标为进行中；SpecForge 是 SGLang 原生备选。详细证据、资源估算和 PoC 门禁见 `docs/autoreply-draft-model-framework-research-2026-08-14.md`。

这部分只有可行性结论，没有 QPS 收益结论。未来任何 draft 必须在同一请求与 8,192 长度约束下重新冷启动二分；只有最大 PASS 边界高于 9.4 QPS 才算有收益，并同时报告绝对 QPS 与百分比。
