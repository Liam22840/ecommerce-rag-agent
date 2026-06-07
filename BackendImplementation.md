# Backend Implementation Notes

本文档说明当前 Basic difficulty 后端的实现逻辑。目标不是只完成一个临时 demo，而是先搭好后续高级功能可以继续扩展的 RAG 后端骨架。

## Scope

当前后端实现的是一个基于商品库的电商导购 API：

- 读取本地商品数据集，生成可返回给前端的商品卡片。
- 由 LLM 解析用户意图和约束（类目、子类目、品牌、价格、卖点、排除项、多轮上下文等），确定性规则做校验和兜底。
- 使用 Milvus Lite 中已有的向量库做语义检索。
- 同时使用本地 lexical retrieval 作为补充和兜底，两路结果用 Reciprocal Rank Fusion 融合。
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
3. `IntentParser` 以 LLM 为主解析筛选条件（确定性规则做校验/兜底），并结合 session 历史补全多轮上下文（承接、相对追问、“换一批”、回看之前商品等）。
4. 按 `intent_type` 路由：闲聊或不在售品类 → chitchat 回复；对比 → 对比决策（见 `ComparisonDecisionAgent.md`）；其余 → 商品检索。
5. `ProductRetriever` 执行 hybrid retrieval（向量为主 + lexical 兜底，RRF 融合）。
6. 软约束（`required_terms`/`requested_specs`）参与排序而不是硬过滤；`excluded_terms`（“不要X”）由 LLM 在候选集上语义判定剔除，确定性否定词匹配兜底。
7. `ProductCatalog` 把命中的商品转换成 `ProductCard`。
8. `build_messages()` 把用户问题、解析结果、候选商品事实，以及诚实信号（`result_status`、未命中的卖点/规格等）组成 prompt。
9. `ArkChatClient.complete()` 调用 Doubao/Ark chat model；不可用时退回确定性 grounded answer。
10. 返回 `ChatResponse`，包含回答、商品卡、解析意图、检索来源和 warning。

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

意图解析在 `server/intent.py`，采用 **LLM 为主、确定性兜底** 的模式（统一的 “LLM 提议 → 确定性校验 → 规则兜底” 架构）：

- **LLM 是主理解器**：把口语、俚语、错别字、中英混写都映射到商品库里的官方类目/子类目/品牌，并理解价格（含中文数字、“三百左右” 这类约数会展开成区间）、卖点、排序偏好、否定、以及多轮上下文。
- **确定性 coercion 做校验**：LLM 的输出会被 clamp 到合法枚举值（类目/子类目/品牌必须真实存在）、价格做数值与区间校验、互相矛盾的项（例如同一品牌既想要又被排除）会被纠正。
- **规则 parser 是兜底**：当 LLM 不可用时，用价格正则、alias 表、品牌子串等规则解析。规则也会在 LLM 在线时补充少量 keyword-extractable 字段（取并集），但不会覆盖 LLM 的判断。

解析结果统一放在 `SearchFilters`：

```python
SearchFilters(
    max_price=None, min_price=None,
    category=None, sub_category=None, brand=None,
    prefer_low_price=False, sort_by="relevance",
    intent_type="product_search",          # product_search | comparison | chitchat
    required_terms=[],                       # 卖点（敏感肌、保湿…），参与排序而非过滤
    requested_specs=[],                      # 规格（50g、256GB…），参与排序而非过滤
    excluded_brands=[], excluded_terms=[],   # 排除：品牌为硬过滤，excluded_terms 由 LLM 在候选集上判定
    compare_refs=[], compare_product_ids=[], # 对比引用 / LLM 解析出的对比商品 id
    recall_product_ids=[],                   # “回到最开始那个” 的回看 id
    exclude_seen=False,                      # “换一批” 的去重标记
    rewritten_query="",                      # 依赖上文时改写成可独立检索的查询
    raw_query="...",
)
```

`SearchFilters` 是 retrieval 和 answer 之间的稳定契约：理解层（LLM/规则）只要继续输出它，下游就不需要大改。原计划里 “把规则 parser 换成 LLM parser” 和 “多轮记忆” 两个扩展点目前都已实现。

## Retrieval Logic

检索在 `server/retrieval.py`，当前是 hybrid retrieval。

### Vector Retrieval

如果以下条件都满足，会启用向量检索：

- `ENABLE_VECTOR_SEARCH=true`
- `ARK_EMBEDDING_API_KEY` 已设置
- `data/milvus.db` 可以打开
- Milvus collection 可用

流程：

1. 使用 `DoubaoEmbedder.embed_text(query)` 把用户 query（或 LLM 改写后的 `rewritten_query`）转成 2048 维向量。
2. 使用 `MilvusStore.search()` 在 `data/milvus.db` 里查 top K chunk，并按 product 去重到每个商品的最佳 chunk。
3. 根据命中的 `product_id` 回到 `ProductCatalog` 取完整商品。
4. 再应用 `SearchFilters` 里的**硬结构化约束**（价格、类目、子类目、品牌、排除品牌）。注意：卖点 `required_terms`、规格 `requested_specs` 不在这里硬过滤（它们只影响排序）。
5. 保留 snippet；该路的相似度只用来给候选排序，最终分数由下面的 RRF 决定。

### Lexical Retrieval

无论 vector 是否成功，都会跑本地 lexical retrieval：

- 根据 query、类目、子类目 alias、品牌、关键词构造 query terms。
- 对 title、brand、category、sub_category、marketing description、FAQ、review 做简单匹配。
- 类目、子类目、品牌、title 命中有更高权重；`required_terms`/`requested_specs` 在这里作为**排序加权**（不是过滤）。
- 硬结构化约束（价格、品牌、类目、排除品牌）先过滤。`excluded_terms` 不在检索阶段过滤，而是检索后由 LLM judge 在候选集上判定剔除。

### Merge and Source（Reciprocal Rank Fusion）

vector 和 lexical 两路各自按本路得分排序后，用 **RRF** 融合，而不是把两路原始分相加：

- 每个商品在某一路的贡献是 `1 / (RRF_K + rank)`（`RRF_K=60`，标准取值），按名次而非原始分大小计分。这样向量余弦和 lexical 词频两套不同量纲不会互相压制（之前是 lexical 量纲偏大，主导排序）。
- 同一个商品被两路命中时，两路贡献相加，`source` 标记为 `hybrid`，snippets 取并集。
- 最终按融合分排序，同分时倾向更低价格。
- 若用户表达了价格/评分偏好（`prefer_low_price`/`sort_by`），会在融合排序之后再按价格或评分重排。

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

测试套件已显著扩展，覆盖 intent（LLM + 规则兜底）、retrieval（含 RRF 融合）、comparison、多轮记忆、排除判定、诚实回复等；`pytest -q` 全部通过。

真实链路 smoke 覆盖：

- `data/milvus.db` 可打开
- Milvus collection row count 是 `1092`
- embedding API 返回 2048 维向量
- `/api/chat` 返回 `retrieval_source=hybrid`
- `/api/chat/stream` 返回 UTF-8 正常的 SSE

## Extension Points

已实现（早期文档里曾列为扩展点）：

- LLM parser：`IntentParser` 已从规则解析升级为 LLM 为主、规则兜底，输出仍是 `SearchFilters`。
- Multi-turn memory：已用 `session_id` 维护对话状态（承接上文、相对追问、“换一批”、回看之前商品）。
- Hybrid fusion：两路检索改为 RRF 融合。

仍可继续扩展：

- Better ranking: 在 `ProductRetriever` 里加入 reranker、用户画像、销量、评分或库存权重（目前为 RRF，无独立 reranker）。
- Multimodal query: 当前 embedding client 已支持 multimodal endpoint，后续可以让用户上传图片并复用同一 embedding/retrieval 管线。
- Cart/actions: 现有 product card 已有稳定 product id 和 detail path，可继续加 add-to-cart、compare、favorite 等 action。
- Observability: `retrieval_source`、`warnings`、`degraded` 已经能暴露基础运行状态，后续可接日志、trace 和 evaluation。
