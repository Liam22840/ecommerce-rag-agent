# 电商智能导购助手

这是一个基于检索增强生成的多模态电商导购系统。用户可以用文字、语音或图片表达购物需求，系统会在真实商品库中检索商品，给出有依据的推荐、对比、加购和模拟下单结果。价格、规格、库存和订单状态都由后端结构化数据核对，模型只负责理解用户意图和组织自然语言。

## 项目内容

- `ecommerce_agent_dataset/`：四个类目的一百余个商品，包含标题、品牌、类目、规格价格、描述、问答、评价和主图。
- `ingestion/`：商品切块、豆包多模态向量生成、Milvus Lite 入库和向量缓存。
- `server/`：FastAPI 后端，包含检索、意图识别、路由、对比、购物车、订单、规划器、图搜和语音播报接口。
- `client/ios/`：SwiftUI 原生 iOS 应用，包含流式聊天、商品卡片、商品详情、收藏、拍照找货、普通话语音输入、语音播报、购物车和订单确认。
- `docs/`：提交说明、运行手册、架构设计、评测用例和各模块设计文档。

## 文档入口

评审或答辩时建议先看提交版文档，再按需要看模块细节。

| 用途 | 文档 |
| --- | --- |
| 提交文档入口 | [`docs/Submission.md`](docs/Submission.md) |
| 项目文档 | [`docs/ProjectDocument.md`](docs/ProjectDocument.md) |
| 设计文档 | [`docs/DesignDocument.md`](docs/DesignDocument.md) |
| 说明文档 | [`docs/UsageDocument.md`](docs/UsageDocument.md) |
| 部署、启动和体验流程 | [`docs/Runbook.md`](docs/Runbook.md) |
| 系统架构细节 | [`docs/Architecture.md`](docs/Architecture.md) |
| 黑箱测试文案和预期结果 | [`docs/EvaluationCases.md`](docs/EvaluationCases.md) |
| 系统总览 | [`docs/Overview.md`](docs/Overview.md) |
| 数据工程与检索 | [`docs/RetrievalAndData.md`](docs/RetrievalAndData.md) |
| 意图、路由和防跑偏 | [`docs/IntentAndRouting.md`](docs/IntentAndRouting.md) |
| 后端接口、流式输出和缓存 | [`docs/BackendService.md`](docs/BackendService.md) |
| iOS 原生体验 | [`docs/ios-ui-design.md`](docs/ios-ui-design.md) |
| 购物车与下单 | [`docs/CartCheckoutAgent.md`](docs/CartCheckoutAgent.md) |
| 对比决策 | [`docs/ComparisonDecisionAgent.md`](docs/ComparisonDecisionAgent.md) |
| 多步任务规划 | [`docs/PlannerAgent.md`](docs/PlannerAgent.md) |

## 环境准备

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入对话、向量和语音播报所需的密钥。
```

后端支持这些主要配置：

| 配置 | 说明 |
| --- | --- |
| `CHAT_API_KEY` / `ARK_CHAT_API_KEY` | 对话模型密钥。 |
| `CHAT_BASE_URL` / `ARK_CHAT_BASE_URL` | OpenAI 兼容对话接口地址。 |
| `CHAT_MODEL` / `ARK_CHAT_MODEL` | 对话模型名称。 |
| `ARK_EMBEDDING_API_KEY` | 豆包多模态向量模型密钥。 |
| `ENABLE_VECTOR_SEARCH` | 是否启用向量检索；关闭后退回关键词检索。 |
| `ENABLE_LLM` | 是否启用模型回答；关闭后使用确定性兜底回答。 |
| `ENABLE_LLM_INTENT` | 是否启用模型意图识别。 |
| `ENABLE_TTS` | 是否启用后端语音播报接口。 |
| `TTS_API_KEY` / `GEMINI_API_KEY` | 语音播报模型密钥。 |

完整配置见 [`.env.example`](.env.example) 和 [`docs/Architecture.md`](docs/Architecture.md)。

## 生成向量库

快速验证 2 个商品：

```bash
.venv/bin/python ingest.py --limit 2
```

完整入库：

```bash
.venv/bin/python ingest.py
```

输出位置：

- `data/milvus.db`：Milvus Lite 向量库。
- `data/embedding_cache.jsonl`：向量缓存，重复运行会复用。

命令结束时会打印 `Milvus collection row count: N`，表示入库完成。

## 启动后端

真实体验建议全开模型、向量和语音：

```bash
ENABLE_VECTOR_SEARCH=true ENABLE_LLM=true ENABLE_LLM_INTENT=true ENABLE_TTS=true \
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

本地无密钥时也可以验证确定性链路：

```bash
ENABLE_VECTOR_SEARCH=false ENABLE_LLM=false ENABLE_TTS=false \
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

核心接口：

- `GET /health`
- `POST /api/chat`
- `POST /api/chat/stream`
- `POST /api/v1/chat/stream`
- `GET /api/products/{product_id}`
- `POST /api/tts`

接口验证：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"推荐一款适合油皮的洗面奶"}'
```

如果向量模型或向量库不可用，系统会降级到关键词检索和确定性回答；不会编造商品事实。

## 启动 iOS 应用

使用 Xcode 打开：

```bash
open client/ios/EcommerceGuideApp/EcommerceGuideApp.xcodeproj
```

Swift Package 当前最低平台是 iOS 17 / macOS 14，需要使用支持该平台的 Xcode。

`EcommerceGuideApp` scheme 默认通过 `SSEChatService` 连接后端。scheme 里可以配置：

- `ECOMMERCE_GUIDE_BACKEND_URL=http://127.0.0.1:8000/api/chat/stream`
- `ECOMMERCE_GUIDE_SERVICE=mock`：需要离线演示时才切到本地 mock。
- `ECOMMERCE_GUIDE_TTS_URL=http://127.0.0.1:8000/api/tts`：需要单独指定语音播报接口时使用。

完整步骤见 [`docs/Runbook.md`](docs/Runbook.md)。

## 运行测试

```bash
.venv/bin/python -m pytest -v
cd client/ios/EcommerceGuide
swift test
```

常用黑箱测试文案和预期结果见 [`docs/EvaluationCases.md`](docs/EvaluationCases.md)。
