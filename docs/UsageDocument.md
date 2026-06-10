# 说明文档

## 环境和依赖

| 项目 | 要求 |
| --- | --- |
| 操作系统 | macOS，建议配合 Xcode 运行 iOS Simulator。 |
| Python | 建议 3.12。 |
| iOS | Swift Package 使用 iOS 17 / macOS 14 作为最低平台。 |
| 后端依赖 | 见 `requirements.txt`。 |
| iOS 依赖 | Swift Package，无额外第三方包。 |
| 向量库 | Milvus Lite，默认读取 `data/milvus.db`。 |

Python 主要依赖：

| 依赖 | 用途 |
| --- | --- |
| `fastapi`、`uvicorn` | 后端接口和服务运行。 |
| `pydantic` | 请求和响应结构校验。 |
| `pymilvus`、`milvus-lite` | 向量库读写。 |
| `requests`、`httpx` | 模型和接口测试。 |
| `python-dotenv` | 读取 `.env` 配置。 |
| `pytest`、`pytest-mock` | 后端测试。 |

## 配置说明

复制配置文件：

```bash
cp .env.example .env
```

真实演示建议填写：

```bash
CHAT_API_KEY=...
CHAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
CHAT_MODEL=gemini-3.1-flash-lite

ARK_EMBEDDING_API_KEY=...
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215

ENABLE_VECTOR_SEARCH=true
ENABLE_LLM=true
ENABLE_LLM_INTENT=true

ENABLE_TTS=true
TTS_API_KEY=...
TTS_MODEL=gemini-3.1-flash-tts-preview
TTS_VOICE=Sulafat
```

配置表：

| 配置 | 说明 |
| --- | --- |
| `CHAT_API_KEY` / `ARK_CHAT_API_KEY` | 对话模型密钥。 |
| `CHAT_BASE_URL` / `ARK_CHAT_BASE_URL` | OpenAI 兼容对话接口地址。 |
| `CHAT_MODEL` / `ARK_CHAT_MODEL` | 对话模型名称。 |
| `ARK_EMBEDDING_API_KEY` | 豆包多模态向量模型密钥。 |
| `ENABLE_VECTOR_SEARCH` | 是否启用向量检索。 |
| `ENABLE_LLM` | 是否启用模型回答。 |
| `ENABLE_LLM_INTENT` | 是否启用模型意图识别和路由。 |
| `ENABLE_QUERY_CACHE` | 是否启用原文查询缓存。 |
| `ENABLE_FILTER_CACHE` | 是否启用结构化条件缓存。 |
| `ENABLE_TTS` | 是否启用 `/api/tts`。 |
| `TTS_API_KEY` / `GEMINI_API_KEY` | 语音播报模型密钥。 |
| `TTS_MODEL` | 语音播报模型。 |
| `TTS_VOICE` | 语音播报音色。 |

## 启动后端

安装依赖：

```bash
uv venv
uv pip install -r requirements.txt
```

真实质量测试使用全开模式：

```bash
ENABLE_VECTOR_SEARCH=true ENABLE_LLM=true ENABLE_LLM_INTENT=true ENABLE_TTS=true \
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

无密钥时可以用降级模式验证主流程：

```bash
ENABLE_VECTOR_SEARCH=false ENABLE_LLM=false ENABLE_TTS=false \
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

如果商品数据发生变化，重建向量库：

```bash
.venv/bin/python ingest.py
```

## 启动 iOS 应用

打开 Xcode 项目：

```bash
open client/ios/EcommerceGuideApp/EcommerceGuideApp.xcodeproj
```

选择 `EcommerceGuideApp` scheme 和 iPhone Simulator。真实后端地址建议配置为：

```text
ECOMMERCE_GUIDE_BACKEND_URL=http://127.0.0.1:8000/api/chat/stream
```

默认使用真实 SSE 后端。只有需要离线演示时才设置：

```text
ECOMMERCE_GUIDE_SERVICE=mock
```

如果 TTS 使用单独地址，可以设置：

```text
ECOMMERCE_GUIDE_TTS_URL=http://127.0.0.1:8000/api/tts
```

## 后端接口

| 接口 | 方法 | 用途 |
| --- | --- | --- |
| `/health` | `GET` | 存活探测。 |
| `/api/chat` | `POST` | 一次性返回完整聊天结果。 |
| `/api/chat/stream` | `POST` | SSE 流式聊天。 |
| `/api/v1/chat/stream` | `POST` | 兼容旧客户端的 SSE 路径。 |
| `/api/products/{product_id}` | `GET` | 商品详情。 |
| `/assets/products/...` | `GET` | 商品图片资源。 |
| `/api/tts` | `POST` | 文本转语音，返回 WAV 音频。 |

普通聊天示例：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"推荐一款适合油皮的洗面奶"}'
```

流式聊天示例：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我推荐跑鞋，对比最便宜的两双，然后把更适合日常跑步的加入购物车"}'
```

TTS 示例：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"我找到一款适合你的商品。"}' \
  --output /tmp/ecommerce-guide-tts.wav
```

## SSE 事件

| 事件 | 客户端表现 |
| --- | --- |
| `token` | 追加到助手气泡。 |
| `products` | 展示商品卡片。 |
| `comparison` | 展示结构化对比卡。 |
| `cart` | 更新购物车状态。 |
| `order` | 展示订单草稿、提交或取消状态。 |
| `plan` | 展示或更新多步计划。 |
| `done` | 结束本轮响应。 |

## 测试命令

后端测试：

```bash
.venv/bin/python -m pytest -v
```

iOS Swift Package 测试：

```bash
cd client/ios/EcommerceGuide
swift test
```

重点测试文件：

| 能力 | 测试文件 |
| --- | --- |
| 商品切块和向量 | `tests/test_chunk.py`、`tests/test_embed.py`、`tests/test_ingest.py` |
| 意图识别和路由 | `tests/test_intent.py`、`tests/test_intent_llm.py`、`tests/test_assistant.py` |
| 检索和缓存 | `tests/test_retrieval.py`、`tests/test_query_cache.py`、`tests/test_cache.py` |
| 对比决策 | `tests/test_comparison.py`、`tests/test_comparison_llm.py`、`tests/test_comparison_evaluation.py` |
| 购物车和订单 | `tests/test_commerce_flow.py` |
| 多步 Planner | `tests/test_planner_flow.py` |
| 后端接口和 SSE | `tests/test_server_app.py` |
| TTS | `tests/test_tts.py` |
| iOS 流程、收藏和骨架屏 | `client/ios/EcommerceGuide/Tests/EcommerceGuideTests/` |

## 黑箱验收用例

| 场景 | 用户输入 | 预期结果 |
| --- | --- | --- |
| 基础推荐 | 推荐一款适合油皮的洗面奶 | 返回库内洁面商品和推荐理由。 |
| 条件筛选 | 200 元以下的蓝牙耳机有哪些？ | 返回预算内耳机，不超过 200 元。 |
| 主动澄清 | 推荐一款手机 | 先问预算、拍照、续航等方向。 |
| 排除约束 | 推荐防晒霜，但不要含酒精的，也不要日系品牌 | 剔除明确不符合条件的商品。 |
| 对比决策 | 第一个和第二个哪个更便宜？ | 明确具体 SKU 价格和更便宜者。 |
| 购物车 | 把第一个加入购物车 | 最近展示的第一个商品加入购物车。 |
| 商品详情和收藏 | 打开第一款商品详情，收藏后进入收藏页 | 详情页展示图片、推荐理由和规格；收藏页能看到该商品。 |
| 下单 | 下单 | 返回待确认订单，不直接提交。 |
| 订单成功 | 确认 | 提交订单，清空购物车，并展示订单成功页。 |
| Planner | 推荐跑鞋，对比最便宜的两双，然后把更适合日常跑步的加入购物车 | 展示计划并逐步执行。 |
| 拍照找货 | 上传图片并说“我想要同款外套” | 返回相似服饰商品。 |
| 语音输入 | 用麦克风说“推荐一款适合干皮的面霜” | 识别后走正常推荐流程。 |

更完整的用例在仓库 `docs/EvaluationCases.md` 中维护。

## 常见问题

| 问题 | 处理方式 |
| --- | --- |
| 8000 端口占用 | 用 `lsof -i :8000` 找旧进程，或换端口并同步 iOS URL。 |
| iOS 秒回但不像真实模型 | 检查是否设置了 `ECOMMERCE_GUIDE_SERVICE=mock`。 |
| iOS 连不上后端 | Simulator 用 `127.0.0.1`；真机用电脑局域网 IP。 |
| iOS 编译提示 macOS 平台过低 | 当前 Swift Package 最低平台是 macOS 14，需要使用支持该平台的 Xcode。 |
| 图片不显示 | 检查后端是否运行，以及 backend URL 是否指向同一台机器。 |
| 拍照按钮不可用 | Simulator 没有真实相机，可用相册；真机可拍照。 |
| 语音输入失败 | 检查系统语音识别和麦克风权限。 |
| TTS 返回 503 | 检查 `TTS_API_KEY` 或 `GEMINI_API_KEY`。 |
| 向量检索失败 | 检查 `data/milvus.db` 和 `ARK_EMBEDDING_API_KEY`；系统会降级关键词检索。 |

## 已知边界

- 库存是演示用合成库存。
- 下单是模拟订单，不接真实支付和物流。
- 商品数据规模是一百余个商品。
- 没有 TTS 密钥时，iOS 会使用系统语音兜底。
