# Backend Implementation Notes

本文档说明当前 Basic difficulty 后端的实现逻辑。目标不是只完成一个临时 demo，而是先搭好后续高级功能可以继续扩展的 RAG 后端骨架。

## Scope

当前后端实现的是一个基于商品库的电商导购 API：

- 读取本地商品数据集，生成可返回给前端的商品卡片。
- 使用规则解析用户意图和约束，例如类目、子类目、品牌、价格上限、排除项。
- 使用 Milvus Lite 中已有的向量库做语义检索。
- 同时使用本地 lexical retrieval 作为补充和兜底。
- 把检索到的商品事实交给 Ark chat model 生成导购回答。
- 当 embedding、Milvus 或 LLM 不可用时，服务不直接崩溃，而是降级到可解释的本地结果。

## Main Request Flow

入口在 `server/app.py`。

### `POST /api/chat`

请求体由 `ChatRequest` 校验：

```json
{
  "message": "推荐一个适合敏感肌的保湿护肤品",
  "session_id": "optional-session-id",
  "top_k": 3
}
```

处理流程：

1. FastAPI 校验 message 非空。
2. `ShoppingAssistant.answer()` 接管请求。
3. `IntentParser` 从用户文本里解析筛选条件。
4. `ProductRetriever` 执行 hybrid retrieval。
5. `ProductCatalog` 把命中的商品转换成 `ProductCard`。
6. `build_messages()` 把用户问题、解析结果和候选商品事实组成 prompt。
7. `ArkChatClient.complete()` 调用 Doubao/Ark chat model。
8. 返回 `ChatResponse`，包含回答、商品卡、解析意图、检索来源和 warning。

### `POST /api/chat/stream`

流式接口复用同一套 prepare 逻辑，只是最后调用 `ArkChatClient.stream()`。返回格式是前端兼容的 SSE：

- `token`: LLM 流式 token。payload 同时包含 `token`、`delta` 和 `text`，兼容不同前端 parser。
- `products`: 最终商品卡片列表。payload 同时包含 `products` 和 `items`，商品字段同时包含 `price/base_price` 和 `matched_reason/reason`。
- `done`: 流结束标记，并附带 `session_id`、`retrieval_source`、`warnings` 等调试信息。

SSE 的 content type 是 `text/event-stream; charset=utf-8`，并且 stream parser 使用 UTF-8 解码，避免中文流式输出乱码。早期版本曾发送 `meta` 和 `delta` 事件；当前版本改为 `token/products/done`，是为了兼容 Liam 的 iOS 前端和旧构建缓存。

### `GET /api/products/{product_id}`

根据 product id 返回原始商品详情。前端商品卡片里的 `detail_path` 会指向这个接口。

## Configuration

配置在 `server/config.py`，从 `.env` 加载。`.env` 已被 `.gitignore` 忽略，真实 key 不会进入 git。

当前 key 分开管理：

```env
ARK_CHAT_API_KEY=
ARK_CHAT_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_CHAT_MODEL=ep-20260514111645-lmgt2

ARK_EMBEDDING_API_KEY=
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
```

这样做的原因是 chat model 和 embedding model 可以使用不同的模型、endpoint 或 key。后续如果替换模型，不需要改业务代码，只需要改配置。

相关开关：

```env
ENABLE_VECTOR_SEARCH=true
ENABLE_LLM=true
```

这两个开关用于本地调试和降级测试。例如没有 LLM key 时仍可以验证 retrieval 和 fallback answer。

## Product Catalog
  
商品数据由 `ProductCatalog.load()` 从 `ecommerce_agent_dataset/*/data/*.json` 读取。

`ProductCatalog` 负责：

- 加载和校验商品 JSON。
- 维护 product id 到商品对象的索引。
- 提供 category、sub_category、brand 集合给 intent parser。
- 根据筛选条件判断商品是否匹配。
- 计算最低 SKU 价格。
- 生成前端使用的 `ProductCard`。
- 生成交给 LLM 的 grounded product facts。
- 提供本地 lexical search。

LLM 不直接拿完整原始 JSON，而是拿经过 `product_facts()` 裁剪后的事实字段，减少 prompt 噪音，并降低模型编造空间。

## Intent Parsing

意图解析在 `server/intent.py`。当前 Basic 版本使用规则解析，这是有意保守的选择：

- 价格解析可确定，例如 “1000以内”、“不超过500”、“200元以上”。
- 类目和子类目使用 alias 表匹配，例如 “跑鞋” 映射到 “跑步鞋”，“降噪耳机” 映射到 “真无线耳机”。
- 子类目可以反推大类，例如 “跑步鞋” 反推 “服饰运动”。
- 品牌从商品库中已有 brand 集合匹配。
- 支持简单排除条件，例如 “不要某品牌”、“不含某成分”。

解析结果统一放在 `SearchFilters`：

```python
SearchFilters(
    max_price=None,
    min_price=None,
    category=None,
    sub_category=None,
    brand=None,
    excluded_brands=[],
    excluded_terms=[],
    raw_query="..."
)
```

这个结构是后续扩展点。以后可以把规则 parser 替换为 LLM/NLU parser，只要继续输出 `SearchFilters`，retrieval 和 answer 逻辑不需要大改。

## Retrieval Logic

检索在 `server/retrieval.py`，当前是 hybrid retrieval。

### Vector Retrieval

如果以下条件都满足，会启用向量检索：

- `ENABLE_VECTOR_SEARCH=true`
- `ARK_EMBEDDING_API_KEY` 已设置
- `data/milvus.db` 可以打开
- Milvus collection 可用

流程：

1. 使用 `DoubaoEmbedder.embed_text(query)` 把用户 query 转成 2048 维向量。
2. 使用 `MilvusStore.search()` 在 `data/milvus.db` 里查 top K chunk。
3. 根据命中的 `product_id` 回到 `ProductCatalog` 取完整商品。
4. 再应用 `SearchFilters`，避免语义命中但硬条件不满足的商品进入结果。
5. 把 Milvus distance 转成内部 score，并保留 snippet。

### Lexical Retrieval

无论 vector 是否成功，都会跑本地 lexical retrieval：

- 根据 query、类目、子类目 alias、品牌、关键词构造 query terms。
- 对 title、brand、category、sub_category、marketing description、FAQ、review 做简单匹配。
- 类目、子类目、品牌、title 命中有更高权重。
- 价格、品牌、类目、排除词等硬条件会先过滤。

### Merge and Source

vector 和 lexical 的结果按 product id 合并：

- 同一个商品被两路命中时合并 score 和 snippets。
- `source` 会标记为 `hybrid`。
- 最终按 score 排序，并用更低价格作为同分时的倾向。

返回的 `retrieval_source` 可能是：

- `vector`: 只有向量检索产生结果
- `lexical`: 只有本地关键词检索产生结果
- `hybrid`: 两路检索共同参与
- `none`: 没有找到匹配商品

## Embedding and Milvus

embedding client 在 `ingestion/embed.py`，Milvus wrapper 在 `ingestion/milvus_store.py`。

当前默认 embedding model：

```text
doubao-embedding-vision-251215
```

该模型用于：

- ingestion 阶段给商品 text/image chunk 建向量。
- query 阶段给用户问题建向量。

必须保持商品库向量和 query 向量使用同一个 embedding space，否则 Milvus 语义检索会失效。

当前 Milvus collection 是 `products`，主键是 `chunk_id`，核心字段包括：

- `product_id`
- `chunk_type`
- `text`
- `category`
- `sub_category`
- `brand`
- `base_price`
- `embedding`

向量维度是 `2048`，metric 是 `COSINE`。

项目默认不重建 `data/milvus.db`。它被视为团队已经生成好的 populated vector store。正常开发只读取它；只有明确需要重新 ingestion 时才运行 `ingest.py`。

## LLM Grounding

prompt 构造在 `server/prompts.py`。

system prompt 明确要求：

- 只能依据提供的商品事实回答。
- 不能编造商品、价格、库存、优惠券、功效或参数。
- 候选商品不足时要说明没有完全匹配项。
- 价格必须使用商品事实中的 `price` 字段。

user message 是一个 JSON payload，包含：

- `user_query`
- `parsed_filters`
- `candidate_products`
- `instruction`

这种结构让模型更容易区分用户问题、机器解析出的约束和真实商品事实。

LLM client 在 `server/llm.py`，调用 OpenAI-compatible Ark API：

- 非流式：`/chat/completions`
- 流式：`/chat/completions` with `stream=true`
- temperature 固定为 `0.2`，减少随机性

## Degradation and Error Handling

当前实现的关键原则是：单个外部能力失败时，API 尽量返回可解释结果，而不是直接 500。

### Vector Search Failure

以下情况会触发 vector 降级：

- embedding key 缺失
- Milvus DB 打不开
- embedding API 调用失败
- Milvus search 抛错

结果：

- warning 写入 response
- lexical retrieval 继续工作
- `degraded=true`

### LLM Failure

以下情况会触发 LLM 降级：

- `ENABLE_LLM=false`
- chat key 缺失
- chat API 返回错误
- stream 中断或解析异常

结果：

- 非流式接口使用 deterministic fallback answer
- 流式接口按固定 chunk 输出 fallback answer
- warning 中说明 LLM unavailable

Fallback answer 仍然只基于商品库命中结果生成，不会编造商品事实。

## Response Shape

`ChatResponse` 字段：

```python
answer: str
products: list[ProductCard]
session_id: str | None
intent: dict
retrieval_source: "vector" | "lexical" | "hybrid" | "none"
degraded: bool
warnings: list[str]
```

`ProductCard` 字段：

```python
product_id: str
title: str
brand: str
category: str
sub_category: str
price: float
image_path: str
detail_path: str
matched_reason: str | None
```

这套 response 对前端比较稳定：前端可以直接渲染回答、商品卡片、详情入口，也可以根据 `degraded` 和 `warnings` 做调试提示。

## Tested Examples

当前真实测试过的例子：

### Example 1

Input:

```text
推荐一个适合敏感肌的保湿护肤品
```

Observed result:

```text
status=200
retrieval_source=hybrid
degraded=False
warnings=[]
```

返回商品包括：

- 理肤泉特安舒缓修复霜
- 薇诺娜舒敏保湿特护霜
- 薇诺娜极润保湿面膜

### Example 2

Input:

```text
我想买一个适合通勤的降噪耳机，预算1000以内
```

Observed result:

```text
status=200
retrieval_source=none
degraded=False
warnings=[]
```

该 case 没有返回商品，回答会明确说明当前商品库没有找到完全匹配项，而不是硬编商品。

### Example 3

Input:

```text
找一双适合跑步的运动鞋，价格不要太贵
```

Observed result:

```text
status=200
retrieval_source=hybrid
degraded=False
warnings=[]
```

返回商品包括：

- HOKA Clifton 9
- 特步 160X 6.0 PRO
- adidas Ultraboost 5

## Test Commands

完整测试：

```bash
.venv/bin/python -m pytest -q
```

注意：Milvus Lite 测试会在本地临时目录 bind Unix socket。如果在受限 sandbox 里跑，可能因为 `Operation not permitted` 失败；在正常本机权限下运行可以通过。

当前验证结果：

```text
27 passed, 1 warning
```

真实链路 smoke 覆盖：

- `data/milvus.db` 可打开
- Milvus collection row count 是 `1092`
- embedding API 返回 2048 维向量
- `/api/chat` 返回 `retrieval_source=hybrid`
- `/api/chat/stream` 返回 UTF-8 正常的 SSE

## Extension Points

当前实现为后续任务保留了这些扩展点：

- Replace parser: 把规则 `IntentParser` 换成 LLM/NLU parser，但保持输出 `SearchFilters`。
- Better ranking: 在 `ProductRetriever` 里加入 reranker、用户画像、销量、评分或库存权重。
- Multi-turn memory: 使用 `session_id` 维护对话状态，补全用户上一轮约束。
- Multimodal query: 当前 embedding client 已支持 multimodal endpoint，后续可以让用户上传图片并复用同一 embedding/retrieval 管线。
- Cart/actions: 现有 product card 已有稳定 product id 和 detail path，可继续加 add-to-cart、compare、favorite 等 action。
- Observability: `retrieval_source`、`warnings`、`degraded` 已经能暴露基础运行状态，后续可接日志、trace 和 evaluation。
