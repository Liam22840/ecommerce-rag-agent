# Backend Implementation Notes

本文档说明当前 Basic difficulty 后端的实现逻辑。目标不是只完成一个临时 demo，而是先搭好后续高级功能可以继续扩展的 RAG 后端骨架。

## Scope

当前后端实现的是一个基于商品库的电商导购 API：

- 读取本地商品数据集，生成可返回给前端的商品卡片。
- 由 LLM 解析用户意图和约束（类目、子类目、品牌、价格、卖点、排除项、多轮上下文等），确定性规则做校验和兜底。
- 使用 Milvus Lite 中已有的向量库做语义检索。
- 同时使用本地 lexical retrieval 作为补充和兜底，两路结果用 Reciprocal Rank Fusion 融合。
- 把检索到的商品事实交给 chat model 生成导购回答。
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
9. `ChatClient.complete()` 调用 chat model；不可用时退回确定性 grounded answer。
10. 返回 `ChatResponse`，包含回答、商品卡、解析意图、检索来源和 warning。

### `POST /api/chat/stream`

流式接口在生成器内部跑 prepare 逻辑（不再在调用前先跑完），并为了“首屏极速响应”调整了事件顺序。返回格式是前端兼容的 SSE，事件顺序：

1. `token`（开场白）：请求一进来就立刻发送一句简短开场白，此时意图解析和检索都还没开始，所以首 token 时间在毫秒级。开场白由规则解析器即时挑选（命中类目就点名，如“好的，我来帮您找面霜～”；对比就说“对比一下”；认不出就用中性“好的～”，不会误导闲聊）。它只是流式装饰，不会写入存储或非流式 JSON 回答。
2. `products` / `comparison`（卡片优先）：检索一完成就先把商品卡片发出去，不等正文写完。payload 同时包含 `products` 和 `items`，商品字段同时包含 `price/base_price` 和 `matched_reason/reason`。
3. `token`（正文）：随后才流式输出 LLM 正文。payload 同时包含 `token`、`delta` 和 `text`，兼容不同前端 parser。
4. `done`：流结束标记，并附带 `session_id`、`retrieval_source`、`warnings` 等调试信息。

SSE 的 content type 是 `text/event-stream; charset=utf-8`，并且 stream parser 使用 UTF-8 解码，避免中文流式输出乱码。早期版本曾发送 `meta` 和 `delta` 事件；当前版本改为 `token/products/done`，是为了兼容 Liam 的 iOS 前端和旧构建缓存。若 prepare 在生成器内部抛错（开场白已发出，无法再返回 500），会优雅地补一句兜底文案再 `done`，不会让前端挂住。

### `GET /api/products/{product_id}`

根据 product id 返回原始商品详情。前端商品卡片里的 `detail_path` 会指向这个接口。

## Configuration

配置在 `server/config.py`，从 `.env` 加载。`.env` 已被 `.gitignore` 忽略，真实 key 不会进入 git。

当前 key 分开管理：

```env
# Chat model — Gemini Flash-Lite via其 OpenAI 兼容 endpoint。
CHAT_API_KEY=
CHAT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
CHAT_MODEL=gemini-3.1-flash-lite

# Embedding model — Doubao（保持不变：milvus.db 是在该 embedding space 建的）。
ARK_EMBEDDING_API_KEY=
ARK_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision-251215
```

chat model 和 embedding model 各自独立配置（可用不同的模型、endpoint、key）。client 是 OpenAI 兼容的，所以替换 chat 模型只需要改配置，不需要改业务代码。

相关开关：

```env
ENABLE_VECTOR_SEARCH=true
ENABLE_LLM=true
ENABLE_LLM_INTENT=true
ENABLE_QUERY_CACHE=true     # 原样重复问题的精确缓存（见“首屏延迟与缓存”）
ENABLE_FILTER_CACHE=true    # 按解析意图缓存，不同措辞同义命中
```

这几个开关用于本地调试和降级测试。例如没有 LLM key 时仍可以验证 retrieval 和 fallback answer；两个缓存默认开启，调试时可单独关掉。

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

LLM 不直接拿完整原始 JSON，而是拿经过 `product_facts()` 裁剪后的事实字段，减少 prompt 噪音，并降低模型编造空间。为了缩短首 token 时间，`product_facts()` 进一步做了压缩：去掉了每个商品里重复的价格说明（同样的规则在 system prompt 里只写一次），描述、FAQ、评价都裁到几条短摘要，但价格和 SKU 字段全部保留（回答靠它们 grounding）。top_k=5 时回答 prompt 从约 15.5k 字符降到约 7.2k。

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

1. 使用 `DoubaoEmbedder.embed_text(query)` 把用户 query（或 LLM 改写后的 `rewritten_query`）转成 2048 维向量。注意：这个向量通常在意图解析时就已经在后台并行算好了（见“首屏延迟与缓存”里的预热），这里会直接复用那次结果，而不是重新算一遍。
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

## 首屏延迟与缓存（Latency & Caching）

冷启动一条全新查询的关键路径是三次串行的网络往返：意图 LLM（约 1.2s）→ 向量 embedding（冷约 1.75s）→ 回答 LLM 首 token（约 1.1s）。下面几处针对“首屏极速响应”做了优化，目标是让用户尽快看到内容，同时不牺牲 grounding 质量。

- **首句开场白**：流式接口在意图解析和检索开始之前就先发出一句开场白（见 `/api/chat/stream`）。这样首 token 时间是毫秒级，真正的回答在它后面继续拼。开场白由规则解析器即时挑选，认不出就中性回复，不会误导。
- **向量预热（流水线并行）**：`ProductRetriever.prewarm_query()` 在意图 LLM 还在读 query 时，就用线程池在后台开始算 query 向量，并把这次计算存成一个 future。检索阶段不再重新 embedding，而是直接 `await` 这个 future（`_embed_query`）。这样 embedding 被藏在意图调用后面，冷查询能省约 1s。只有当意图 LLM 把 query 改写成不同文本（约三分之一的首轮查询，改写本身有利于检索）时，预热才用不上，退回原来的串行速度，绝不会更慢。embedding 缓存因此会被多线程同时读写，已加锁保证安全。
- **Prompt 压缩**：见 Product Catalog 一节，回答 prompt 约减半。
- **两级回答缓存**：
  - 精确缓存 `QueryCache`（`server/query_cache.py`）：按归一化后的原始 query + top_k 做 key，在 assistant 之前短路，命中 <0.1s。只缓存无上下文的 `product_search`（带对比/最近商品上下文的轮次不缓存）。
  - 意图缓存 `FilterCache`（`server/filter_cache.py`，继承 `QueryCache` 的存储）：按**解析后的 `SearchFilters`** 做 key（只取语义字段，归一化并排序列表字段），所以“便宜的洗面奶 / 平价一点的洁面 / 实惠的洗面奶”这类同义不同写法会命中同一条，而“便宜”和“不便宜”因为解析出的 filters 不同绝不会撞到一起。它在 assistant 内部、`parse()` 之后查（需要 filters 才能算 key），命中可跳过 embedding、检索和回答 LLM，整轮回答在约 1.3s 落地。catalog 是静态的，所以只做 LRU 上限，不设 TTL。

实测（真实模型）：首 token 约 1–6ms；冷启动完整回答约 2.5–3.5s；精确重复 <0.1s；同义改写约 1.3s。检索结果在有无预热下逐字节一致；回答价格/SKU 正确、未命中条件会如实说明、无编造。

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

LLM client 在 `server/llm.py`，调用 OpenAI 兼容的 chat API：

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
price_label: str                      # 照抄给前端/LLM 的价格文案，如 "89元起（15g 体验装）"
price_summary: str                    # 多规格 SKU 价格明细
lowest_price_sku: SkuPrice | None
selected_price_sku: SkuPrice | None   # 命中用户所要规格时的对应 SKU
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
- 首屏极速响应：开场白即时首 token、卡片优先、向量预热并行、prompt 压缩（见“首屏延迟与缓存”）。
- 回答缓存：精确缓存 + 按意图的同义缓存。

仍可继续扩展：

- Better ranking: 在 `ProductRetriever` 里加入 reranker、用户画像、销量、评分或库存权重（目前为 RRF，无独立 reranker）。
- Multimodal query: 当前 embedding client 已支持 multimodal endpoint，后续可以让用户上传图片并复用同一 embedding/retrieval 管线。
- Cart/actions: 现有 product card 已有稳定 product id 和 detail path，可继续加 add-to-cart、compare、favorite 等 action。
- Observability: `retrieval_source`、`warnings`、`degraded` 已经能暴露基础运行状态，后续可接日志、trace 和 evaluation。
