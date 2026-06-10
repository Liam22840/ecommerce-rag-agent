# 部署与体验手册

这份文档给评审和团队成员使用，目标是用最少步骤跑通完整演示：后端服务、iOS Simulator、流式推荐、商品卡、详情收藏、对比、购物车、订单、拍照找货、语音输入和语音播报。

## 前置条件

- macOS 和 Xcode，建议使用支持 iOS 17 和 macOS 14 Swift Package 的 Xcode。
- Python 3.12。
- `uv` 或本地虚拟环境工具。
- 可用的大模型密钥和向量模型密钥。
- 已克隆本仓库。

## 1. 安装后端依赖

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
```

随后编辑 `.env`。真实演示建议至少填写：

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

如果对话模型使用火山 Ark 的 OpenAI 兼容接口，也可以使用：

```bash
ARK_CHAT_API_KEY=...
ARK_CHAT_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_CHAT_MODEL=doubao-seed-2-0-lite
```

## 2. 检查或重建向量库

仓库默认使用 `data/milvus.db`。如果数据被改过，重新入库：

```bash
.venv/bin/python ingest.py
```

快速验证可以只跑两个商品：

```bash
.venv/bin/python ingest.py --limit 2
```

命令结束时应看到：

```text
Milvus collection row count: N
```

## 3. 启动后端

真实质量测试使用全开模式：

```bash
ENABLE_VECTOR_SEARCH=true ENABLE_LLM=true ENABLE_LLM_INTENT=true ENABLE_TTS=true \
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

如果 8000 端口被占用，先找进程：

```bash
lsof -i :8000
```

然后停止旧服务，或改用其他端口：

```bash
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8001
```

改端口后，iOS 的 `ECOMMERCE_GUIDE_BACKEND_URL` 也要同步改成对应端口。

## 4. 验证后端接口

健康检查：

```bash
curl -sS http://127.0.0.1:8000/health
```

一次性聊天：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"推荐一款适合油皮的洗面奶"}'
```

流式聊天：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我推荐跑鞋，对比最便宜的两双，然后把更适合日常跑步的加入购物车"}'
```

语音播报：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，我是你的 AI 购物助手。"}' \
  --output /tmp/ecommerce-guide-tts.wav
```

如果 `/api/tts` 返回 503，通常是 `TTS_API_KEY` 或 `GEMINI_API_KEY` 未配置。iOS 端会降级到系统语音，不影响聊天主流程。

## 5. 启动 iOS Simulator

打开 Xcode 项目：

```bash
open client/ios/EcommerceGuideApp/EcommerceGuideApp.xcodeproj
```

选择 `EcommerceGuideApp` scheme 和一个 iPhone Simulator。默认 scheme 已指向本机后端，开箱即用：

```text
ECOMMERCE_GUIDE_BACKEND_URL=http://127.0.0.1:8000/api/chat/stream
```

Simulator 和后端跑在同一台 Mac 上时，`127.0.0.1` 始终可达，不用改动。只有用真机调试时，才把它改成这台 Mac 的局域网 IP（如 `http://192.168.x.x:8000/api/chat/stream`）。

如果需要离线 mock：

```text
ECOMMERCE_GUIDE_SERVICE=mock
```

不设置 `ECOMMERCE_GUIDE_SERVICE` 时，Host App 默认走真实 SSE 后端。

如果 TTS 要走单独地址，可以补：

```text
ECOMMERCE_GUIDE_TTS_URL=http://127.0.0.1:8000/api/tts
```

## 6. 推荐演示顺序

1. 打开应用，进入对话页。
2. 输入“推荐一款适合油皮的洗面奶”。
3. 观察流式回复、商品骨架屏和商品卡片。
4. 打开第一款商品详情，收藏后从顶部收藏入口确认收藏列表。
5. 输入“200 元以内的蓝牙耳机有哪些？”。
6. 输入“第一个和第二个哪个更便宜？”。
7. 输入“把更便宜的加入购物车”，观察加购状态和飞入购物车动效。
8. 输入“数量改成 2”。
9. 输入“下单”。
10. 在订单卡或订单确认页编辑收货地址，然后点击“确认下单”，观察订单成功页。
11. 进入拍照找货页面，选择或拍摄一张图片，输入“找同款外套”。
12. 点击麦克风，使用普通话说一个商品需求。
13. 等回答结束后，点击语音播放按钮或观察自动播报。

## 7. 常见问题

| 问题 | 处理 |
| --- | --- |
| 后端提示端口占用 | 用 `lsof -i :8000` 找旧进程，或改端口并同步 iOS URL。 |
| iOS 秒回但不是实时模型效果 | 检查是否设置了 `ECOMMERCE_GUIDE_SERVICE=mock`。真实后端应不设置该变量。 |
| iOS 连不上后端 | Simulator 访问本机可用 `127.0.0.1`；真机需要使用电脑局域网 IP。 |
| iOS 编译提示 macOS 平台过低 | 当前 Swift Package 最低平台是 macOS 14，需要使用支持该平台的 Xcode。 |
| 商品图片不显示 | 检查后端是否运行，以及 `ECOMMERCE_GUIDE_BACKEND_URL` 是否指向同一台机器。 |
| 拍照按钮在 Simulator 不可用 | Simulator 没有真实相机，可用相册选择图片；真机可拍照。 |
| 语音输入失败 | 检查系统语音识别和麦克风权限。 |
| TTS 没有云端音色 | 检查 `/api/tts` 和 `TTS_API_KEY`；失败时 iOS 会使用系统语音。 |
| 向量检索失败 | 检查 `data/milvus.db` 和 `ARK_EMBEDDING_API_KEY`；系统会降级关键词检索。 |
| 回答里没有模型风格 | 检查 `ENABLE_LLM=true` 和对话模型密钥。 |

## 8. 测试命令

后端测试：

```bash
.venv/bin/python -m pytest -v
```

iOS Swift Package 测试：

```bash
cd client/ios/EcommerceGuide
swift test
```

重点黑箱用例见 [`EvaluationCases.md`](EvaluationCases.md)。
