# Medical-RAG

## 项目简介
针对医疗影像辅助分析与专业知识问答需求，本项目实现了基于视觉语言模型（VLM）的 Medical-RAG 系统。系统支持上传医学影像与自然语言提问，自动检索权威医学知识并生成循证解答。

## 核心能力
- 多模态混合指令微调：基于自建医学文本与影像（MedVQA）问答指令数据集，使用 LoRA 微调 Qwen3-VL-8B-Instruct。
- 高精度医学知识检索：权威指南切块入库 Milvus，统一通过 vLLM OpenAI 接口调用 Qwen3-VL-Embedding-2B 与 Qwen3-VL-Reranker-2B。
- 多阶段联合推理：多路召回 + 语义精排 + 图文联合推理，提升 RAG 证据质量与图文一致性。

## 目录结构
```text
Medical-RAG/
├── app.py
├── api.py
├── build_index.py
├── config/
│   ├── settings.yaml
│   └── settings.lite.yaml
├── config.py
├── data/
│   └── knowledge/
├── pipeline.py
├── prompts.py
├── query_once.py
├── schemas.py
├── streamlit_app.py
├── scripts/
│   ├── build_index.py
│   ├── download_qwen_models.py
│   ├── query_once.py
│   ├── run_api.py
│   ├── run_app.py
│   ├── run_streamlit.py
│   └── serve_vllm.sh
├── src/medical_rag/
│   ├── app.py
│   ├── api_server.py
│   ├── config.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── retrieval/
│   │   ├── embedding.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── vlm/
│       └── qwen_vl.py
└── requirements.txt
```

## 环境准备
```bash
cd /root/autodl-tmp/Medical/Medical-RAG
pip install -r requirements.txt
```

## 1) 从魔塔下载 Qwen 检索模型
```bash
python scripts/download_qwen_models.py --cache-dir /root/autodl-tmp/Qwen
```

默认下载到：
- `/root/autodl-tmp/Qwen/Qwen/Qwen3-VL-Embedding-2B`
- `/root/autodl-tmp/Qwen/Qwen/Qwen3-VL-Reranker-2B`

若根盘空间不足，可改为下载到内存盘（重启后会丢失）：
```bash
python scripts/download_qwen_models.py --cache-dir /dev/shm/Qwen
```

## 2) 启动 vLLM 服务
`scripts/serve_vllm.sh` 支持三种服务模式：`embed` / `rerank` / `vlm`。

```bash
# Embedding 服务（默认 8001）
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh embed

# Reranker 服务（默认 8002）
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh rerank

# 如果 reranker 下载在 /dev/shm，指定模型路径
CUDA_VISIBLE_DEVICES=0 RERANK_MODEL_PATH=/dev/shm/Qwen/Qwen/Qwen3-VL-Reranker-2B bash scripts/serve_vllm.sh rerank

# VLM 服务（默认 8003，使用已合并 LoRA 的模型）
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_vllm.sh vlm
```

说明：
- 单卡显存不足时，建议分时启动服务；双卡建议使用 `run_dual_gpu_stack.sh` 固定分卡部署。
- `vlm` 走 OpenAI 兼容接口 `/v1/chat/completions`；retrieval 走 `/v1/embeddings` 与 `/v1/rerank`。
- 脚本已对 `embed/rerank` 默认开启 `--enforce-eager`（更稳），如需关闭可设 `EMBED_ENFORCE_EAGER=0` 或 `RERANK_ENFORCE_EAGER=0`。

### 双卡并行部署（推荐）
若为双卡机器，推荐将检索侧与 VLM 分卡部署：
- `GPU0`：`embedding + reranker`
- `GPU1`：`vlm`

可直接使用一键脚本：
```bash
cd /root/autodl-tmp/Medical/Medical-RAG

# 启动（默认 RETRIEVAL_GPU=0, VLM_GPU=1）
bash scripts/run_dual_gpu_stack.sh start

# 查看状态
bash scripts/run_dual_gpu_stack.sh status

# 停止
bash scripts/run_dual_gpu_stack.sh stop
```

可通过环境变量覆盖：
```bash
RETRIEVAL_GPU=1 VLM_GPU=0 API_PORT=9000 bash scripts/run_dual_gpu_stack.sh restart
```

说明：
- `run_dual_gpu_stack.sh` 默认使用“已合并 LoRA”的 VLM 模型目录，并默认开启 `VLM_SKIP_MM_PROFILING=1`。
- 若你要切回“运行时 LoRA 挂载”模式，可显式传入：
```bash
VLM_LORA_PATH=/root/autodl-tmp/Medical/LlamaFactory/saves/Qwen3-VL-8B-Instruct/lora/train_2026-01-22-16-14-48 \
bash scripts/run_dual_gpu_stack.sh restart
```

## 3) 配置说明
默认配置文件：`config/settings.yaml`

- `embedding.provider=api`，默认指向 `http://127.0.0.1:8001/v1/embeddings`
- `reranker.provider=api`，默认指向 `http://127.0.0.1:8002/v1/rerank`
- `vlm.backend=vllm_openai`，默认指向 `http://127.0.0.1:8003/v1/chat/completions`

`vlm` 关键字段：
- `vlm.backend`：`vllm_openai`
- `vlm.model_name`：请求时使用的模型名（需与 vLLM `--served-model-name` 对齐）
- `vlm.api_base_url`：vLLM 服务地址
- `vlm.api_chat_path`：默认 `/v1/chat/completions`

## 4) 知识入库（切块 + 向量化 + Milvus）
```bash
# 顶层入口
python build_index.py --config config/settings.yaml --drop-old

# 或 scripts 入口
python scripts/build_index.py --config config/settings.yaml --drop-old
```

本地 milvus-lite 调试可使用：
```bash
# 顶层入口
python build_index.py --config config/settings.lite.yaml --drop-old

# 或 scripts 入口
python scripts/build_index.py --config config/settings.lite.yaml --drop-old
```

## 5) 启动系统（上传影像 + 提问）
```bash
# 顶层入口
python app.py --config config/settings.yaml

# 或 scripts 入口
python scripts/run_app.py --config config/settings.yaml
```

默认地址：`http://0.0.0.0:7860`

## 6) FastAPI + Streamlit 交互界面（推荐）
先启动后端 API，再启动 Streamlit 前端。
推荐使用 `config/settings.fastapi.yaml`（Milvus 使用本地 lite DB，不依赖独立 Milvus 服务）。

```bash
# 1) 启动 FastAPI 后端（默认 9000）
python api.py --config config/settings.fastapi.yaml --host 0.0.0.0 --port 9000
# 或
python scripts/run_api.py --config config/settings.fastapi.yaml --host 0.0.0.0 --port 9000

# 2) 启动 Streamlit 前端（默认 8501）
python scripts/run_streamlit.py --host 0.0.0.0 --port 8501 --api-base http://127.0.0.1:9000
```

访问地址：
- FastAPI 文档：`http://<服务器IP>:9000/docs`
- Streamlit 页面：`http://<服务器IP>:8501`

命令行单次问答：
```bash
# 顶层入口
python query_once.py --config config/settings.yaml --image /path/to/image.png --question "该影像提示什么异常？"

# 或 scripts 入口
python scripts/query_once.py --config config/settings.yaml --image /path/to/image.png --question "该影像提示什么异常？"
```

## 7) 一键连通性验证
使用校验脚本检查 embedding / rerank / milvus：

```bash
# 仅验证 embedding API（需先启动 embed 服务）
python scripts/validate_stack.py --config config/settings.yaml --checks embed

# 仅验证 rerank API（需先启动 rerank 服务）
python scripts/validate_stack.py --config config/settings.yaml --checks rerank

# 验证 milvus-lite 连通性
python scripts/validate_stack.py --config config/settings.lite.yaml --checks milvus
```

说明：
- 单卡通常无法同时稳定承载 `embed + rerank + vlm` 三个服务，建议分时或多卡部署。
- 若使用外部 API（如魔塔托管）可将本地服务改为远程端点。

## 8) 端到端测试（FastAPI /ask）
在服务启动后，可用 E2E 脚本直接验证“影像上传 -> 检索 -> 重排 -> VLM 生成”全链路：

```bash
cd /root/autodl-tmp/Medical/Medical-RAG

python scripts/e2e_test.py \
  --api-base http://127.0.0.1:9000 \
  --image /root/autodl-tmp/Medical/VQA_data/802fe124-1b5c-11ef-b341-000066532cad.jpg \
  --question "请结合影像和证据，给出主要异常与下一步建议。" \
  --save-json /tmp/medical_rag_e2e_result.json
```

## 推理链路说明
1. 多路召回：对问题做查询扩展并在 Milvus 检索，使用 RRF 融合结果。
2. 语义精排：Reranker 对候选切片重排序，筛出高置信证据。
3. 图文联合推理：将医学影像、问题、证据一并输入 Qwen3-VL（vLLM OpenAI 接口）输出答案。

## 安全声明
- 系统输出仅用于科研与教学，不构成临床诊断结论。
- 高风险病例必须由具备资质的医生完成最终判读与决策。
