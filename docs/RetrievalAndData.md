# 数据与检索 (Data & Retrieval)

对应 spec 3.1「数据工程与特征治理」。本文档说明商品数据如何被切分、向量化、并检索成可用于回答的事实。意图理解与路由见 `IntentAndRouting.md`，API / 流式 / 缓存 / 降级等服务层见 `BackendService.md`。

## 商品库 (Product Catalog)

商品数据由 `ProductCatalog.load()` 从 `ecommerce_agent_dataset/*/data/*.json` 读取。`ProductCatalog` 负责：

- 加载和校验商品 JSON，维护 product id 到商品对象的索引。
- 提供 category、sub_category、brand 集合给 intent parser。
- 根据筛选条件判断商品是否匹配，计算最低 SKU 价格。
- 生成前端使用的 `ProductCard`，以及交给 LLM 的 grounded product facts。
- 提供本地 lexical search。
- 加载时按 `product_id` 确定性生成合成库存（库存是为固定数据集播种的，不是真实仓储；详见 `CartCheckoutAgent.md` 的「库存」节）。

LLM 不直接拿完整原始 JSON，而是拿经过 `product_facts()` 裁剪后的事实字段，减少 prompt 噪音、降低编造空间。为缩短首 token，`product_facts()` 进一步压缩：重复的价格说明只在 system prompt 写一次，描述/FAQ/评价裁到几条短摘要，但价格和 SKU 字段全部保留（回答靠它们 grounding）。top_k=5 时回答 prompt 从约 15.5k 字符降到约 7.2k。

## Chunking（切分）

入库前，每个商品由 `ingestion/chunk.py` 切成若干**带类型的 chunk**，而不是按固定字数滑窗：

- `summary`：标题 + 类目 + 卖点 + marketing_description 合成的一条概要。
- `faq`：每条官方问答各一个 chunk。
- `review`：每条用户评价各一个 chunk（带评分）。
- `image`：商品主图一个 chunk。

每个 chunk 是一个自洽的可检索单元，按语义边界切分，平衡召回与精确度。`chunk_id` 形如 `{product_id}::{suffix}`，入 Milvus 时回到 `product_id` 去重到每个商品的最佳 chunk。

## Embedding 与 Milvus

embedding client 在 `ingestion/embed.py`，Milvus wrapper 在 `ingestion/milvus_store.py`。默认 embedding model：

```text
doubao-embedding-vision-251215
```

该（多模态）模型同时用于 ingestion 阶段给 text / image chunk 建向量，和 query 阶段给用户问题（或上传图片）建向量。**商品库向量和 query 向量必须用同一个 embedding space**，否则语义检索失效。

Milvus collection 是 `products`，主键 `chunk_id`，核心字段：`product_id / chunk_type / text / category / sub_category / brand / base_price / embedding`。向量维度 `2048`，metric `COSINE`，索引 `AUTOINDEX`。

项目默认不重建 `data/milvus.db`，把它当作团队已生成的 populated vector store；正常开发只读取它，只有明确需要重新 ingestion 时才运行 `ingest.py`。

## 检索 (Hybrid Retrieval)

检索在 `server/retrieval.py`，是 hybrid retrieval（向量 + lexical，RRF 融合）。

### 向量检索

满足以下条件才启用：`ENABLE_VECTOR_SEARCH=true`、`ARK_EMBEDDING_API_KEY` 已设置、`data/milvus.db` 可打开、Milvus collection 可用。流程：

1. `DoubaoEmbedder.embed_text(query)` 把 query（或 LLM 改写后的 `rewritten_query`）转成 2048 维向量。注意：这个向量通常在意图解析时就已在后台并行算好（见 `BackendService.md` 的预热），这里直接复用 future，不重复 embedding。
2. `MilvusStore.search()` 查 top K chunk，按 product 去重到每个商品的最佳 chunk。
3. 按命中的 `product_id` 回 `ProductCatalog` 取完整商品。
4. 再应用 `SearchFilters` 的**硬结构化约束**（价格、类目、子类目、品牌、排除品牌）。卖点 `required_terms`、规格 `requested_specs` 不在这里硬过滤（只影响排序）。
5. 保留 snippet；该路相似度只用于候选排序，最终分数由 RRF 决定。

### lexical 检索

无论 vector 是否成功，都跑本地 lexical retrieval：按 query / 类目 / 子类目 alias / 品牌 / 关键词构造 query terms，对 title、brand、category、sub_category、marketing description、FAQ、review 做匹配。类目/子类目/品牌/title 命中权重更高；`required_terms`/`requested_specs` 在这里是**排序加权**而非过滤。硬结构化约束先过滤；`excluded_terms` 不在检索阶段过滤，而是检索后由 LLM judge 在候选集上判定剔除。

### RRF 融合

vector 和 lexical 两路各自按本路得分排序后，用 **Reciprocal Rank Fusion** 融合，而不是把两路原始分相加：

- 每个商品在某一路的贡献是 `1 / (RRF_K + rank)`（`RRF_K=60`），按名次而非原始分大小计分，避免余弦和词频两套量纲互相压制。
- 同一商品被两路命中时贡献相加，`source` 记为 `hybrid`，snippets 取并集。
- 按融合分排序，同分倾向更低价格；若用户表达了价格/评分偏好（`prefer_low_price`/`sort_by`），在融合排序之后再按价格或评分重排。

`retrieval_source` 取值：`vector`（只有向量出结果）/ `lexical`（只有关键词出结果）/ `hybrid`（两路共同参与）/ `none`（没找到匹配）。

## 数据一致性

价格、SKU 等关键参数全部来自结构化字段（catalog → `ProductCard`/SKU → `pricing.py`），从不从 LLM 文案里取，所选 SKU 的价格也会随购物车回传保持一致。库存这一项数据集没有，是合成的，但它的台账、加购强制和下单扣减是真的（详见 `CartCheckoutAgent.md` 的「库存」节）。
