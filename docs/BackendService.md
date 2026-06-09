# 后端服务 (Backend Service)

本文档涵盖后端服务化：API、流式接口、配置、首屏延迟与缓存、降级容错、返回结构。意图与路由见 `IntentAndRouting.md`，数据与检索见 `RetrievalAndData.md`，各 Agent 见对应文档。入口在 `server/app.py`。

## 请求流程 (Main Request Flow)

### `POST /api/chat`（非流式）

请求体由 `ChatRequest` 校验，除 `message / session_id / top_k` 外，`client_context` 还带 `cart_items`、`recent_product_ids`、`compare_product_ids`、`address`，以及可选 `attachments`（图片）。处理流程：

1. FastAPI 校验 message 非空。
2. `ShoppingAssistant.answer()` 接管。
3. LLM 路由分类器判类别（带 `last_route` 等上下文，见 `IntentAndRouting.md`）；带图片的请求在路由前就走图搜。
4. 按路由分派：chitchat 直接回短句；comparison / cart / checkout / plan 各走对应 Agent；search 走检索。只有 search / comparison 跑较重的意图解析。
5. 检索 → 软约束排序 / `excluded_terms` 由 LLM 在候选集上剔除 → 转 `ProductCard`。
6. `build_messages()` 组 prompt（用户问题 + 解析结果 + 候选事实 + 诚实信号），`ChatClient.complete()` 生成回答，不可用时退确定性兜底。
7. 返回 `ChatResponse`。

### `POST /api/chat/stream`（流式，另有别名 `/api/v1/chat/stream`）

流式接口在生成器内部跑 prepare，并为“首屏极速响应”调整了事件顺序。SSE，事件顺序：

1. `token`（开场白）：请求一进来先发一句中性短开场白（`opener_lead`，如“好的，”），此时路由都还没跑，首 token 在毫秒级；路由判定后再补一句和类别相关的尾巴（`opener_continuation`，如“我来帮您找面霜～”“对比一下”，chitchat 没有尾巴、由内联回复自报）。开场白只是流式装饰，不写入存储或非流式 JSON；它是延迟手段而非路由，猜错只是多/少一句开场白，不改变实际路由。
2. `plan`：多步规划轮在 `prepare_stream` 阶段就逐步发 `plan` 帧（循环前一帧、每步置 running 一帧、done/failed 再一帧），所以 plan 帧先于下面的卡片。
3. `cart` / `order`：购物车 / 下单轮发对应帧，且在卡片**之前**发——`cart` 购物车状态，`order` 订单状态（含金额与收货地址）。`order` 帧的 `type` 为 `order_submitted`（已提交）或 `order_draft`（待确认与已取消都归这个）。
4. `products` / `comparison`（卡片）：检索一完成就把卡片发出去，不等正文。payload 同时含 `products`/`items`，商品字段同时含 `price/base_price` 和 `matched_reason/reason`。
5. `token`（正文）：随后流式输出正文，payload 同时含 `token`/`delta`/`text` 兼容不同前端。
6. `done`：结束标记，附 `session_id`、`retrieval_source`、`warnings`。

content type 是 `text/event-stream; charset=utf-8`，UTF-8 解码避免中文乱码。当前发送 `token / products / comparison / cart / order / plan / done`。若 prepare 在生成器内抛错（开场白已发出，无法再返 500），会优雅补一句兜底再 `done`，不让前端挂住。

### `GET /api/products/{product_id}`

按 product id 返回原始商品详情，前端卡片的 `detail_path` 指向它。

### 其余端点

`GET /health` 返回 `{"status": "ok"}` 做存活探测；`/assets/products` 用 `StaticFiles` 挂在数据集目录上，给商品卡的 `image_path` 提供图片。

## 配置 (Configuration)

配置在 `server/config.py`，从 `.env` 加载（`.env` 已被 gitignore，真实 key 不进 git）。chat model 与 embedding model 各自独立配置：

```env
# Chat model — OpenAI 兼容 endpoint。
CHAT_API_KEY=
CHAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
CHAT_MODEL=gemini-3.1-flash-lite

# Embedding model — Doubao（保持不变：milvus.db 是在该 embedding space 建的）。
ARK_EMBEDDING_API_KEY=
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
```

client 是 OpenAI 兼容的，换 chat 模型只改配置不改业务代码。降级/调试开关：`ENABLE_VECTOR_SEARCH` / `ENABLE_LLM` / `ENABLE_LLM_INTENT` / `ENABLE_QUERY_CACHE` / `ENABLE_FILTER_CACHE`。另有若干 env-overridable 运营旋钮（如单行数量上限 `RAG_MAX_CART_QUANTITY`、各类 cap）。

## 首屏延迟与缓存 (Latency & Caching)

冷启动一条全新查询的关键路径是三次串行往返：意图 LLM（约 1.2s）→ 向量 embedding（冷约 1.75s）→ 回答 LLM 首 token（约 1.1s）。优化：

- **首句开场白**：见上，首 token 毫秒级。
- **首句开场白（图搜）**：图搜轮的 VLM embedding 更慢，单独先发一句 `photo_opener` 占住首屏，道理同上。
- **向量预热（流水线并行）**：`ProductRetriever.prewarm_query()` 在意图 LLM 还在读 query 时就用线程池后台算 query 向量并存成 future，检索阶段直接 `await`，冷查询省约 1s。只有搜索路径用得上：带 `compare_product_ids` 的对比轮不预热。只有当意图 LLM 把 query 改写成不同文本时预热用不上，退回串行速度，绝不会更慢。embedding 缓存多线程读写已加锁。
- **Prompt 压缩**：回答 prompt 约减半（见 `RetrievalAndData.md` 的 product facts 压缩）。
- **两级回答缓存**：精确缓存 `QueryCache`（按归一化 query + top_k，命中 <0.1s，只缓存无上下文的 `product_search`）；意图缓存 `FilterCache`（按解析后的 `SearchFilters` 语义字段，同义不同写法命中同一条，命中跳过 embedding/检索/回答 LLM，约 1.3s）。catalog 静态，只设 LRU 上限不设 TTL。
- **不走缓存的轮次**：带购物车/对比/最近商品上下文、或 commerce/planner/attachments 的轮次；此外只要该 session 已有历史（`has_session_history`），这一轮可能是依赖上文的改写（“便宜点的”），文本缓存既不命中也不写入，以服务端会话记忆为准。`FilterCache` 在 `exclude_seen`（“换一批”）/ `recall_product_ids`（“回到最开始”）时也不缓存，避免把翻新 / 回看的请求错误复用。
- **缓存命中也登记会话记忆**：命中后用 `record_cached_turn` 从缓存响应里重建“这轮展示过的商品 + filters”写回会话记忆，让接下来的指代追问仍能正确解析；流式命中走 `_sse_replay`，并补一句和新轮一样的开场白，让缓存回复读起来和现算的一致。

## 降级与容错 (Degradation)

关键原则：单个外部能力失败时，API 尽量返回可解释结果，而不是直接 500。

- **向量检索失败**（embedding key 缺失 / Milvus 打不开 / API 失败 / search 抛错）：写 warning，`degraded=true`，lexical retrieval 继续工作。图片 embedding 失败也降级到 lexical。
- **LLM 失败**（`ENABLE_LLM=false` / chat key 缺失 / API 错误 / stream 中断）：非流式用确定性兜底回答，流式按固定 chunk 输出兜底，warning 说明 LLM 不可用。兜底回答仍只基于命中结果，不编造。
- **路由 LLM 失败**：退回关键词路由（见 `IntentAndRouting.md`）。

## 返回结构 (Response Shape)

`ChatResponse`：

```python
answer: str
products: list[ProductCard]
comparison: ProductComparison | None   # 对比结果（维度行 + winner）
cart: CartUpdate | None                # 购物车状态
order: OrderDraft | None               # 订单状态（待确认/已提交/已取消，含地址）
plan: ExecutionPlan | None             # 多步计划及每步状态
session_id: str | None
intent: dict
retrieval_source: "vector" | "lexical" | "hybrid" | "none"
degraded: bool
warnings: list[str]
```

`ProductCard`：`product_id / title / brand / category / sub_category / price / price_label / price_summary / lowest_price_sku / selected_price_sku / image_path / detail_path / matched_reason`。前端可直接渲染回答、卡片、详情入口，并按 `degraded` / `warnings` 做调试提示。

## 测试

```bash
.venv/bin/python -m pytest -q
```

测试覆盖意图（LLM + 规则兜底）、检索（含 RRF）、对比、多步规划、购物车与下单、库存、收货地址、多轮记忆、排除判定、诚实回复、路由分类等。注意：Milvus Lite 测试在受限 sandbox 里可能因 `Operation not permitted` 失败，正常本机权限下可通过。

## 现状与后续

已上线：LLM 意图解析、多轮记忆、hybrid RRF 检索、首屏极速响应（开场白 + 预热 + 卡片优先 + prompt 压缩）、两级回答缓存、图搜、购物车与下单、多步规划、库存、上下文路由。后续可继续：检索 reranker / 评分权重、真实支付与物流、把会话内购物车/订单/库存状态持久化到数据库、日志 trace 与 evaluation。
