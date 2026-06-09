# Comparison Decision Agent

本文档说明“对比决策”功能目前的实现策略、请求流程、数据结构使用方式，以及可复制的调试测试文案。

## 功能目标

对比决策用于回答这类问题：

- “A 和 B 哪个更保湿？”
- “第一个和第二个哪个更便宜？”
- “这两双跑鞋穿一天哪个不累？”
- “这两个耳机通勤地铁里哪个更安静，戴久了也舒服？”

核心要求：

- 先确定要对比的商品 ID。
- 价格、SKU、规格等硬事实必须来自结构化字段。
- 用户关心的体验维度由 LLM 抽取，后端再基于商品库证据检索和打分。
- 如果证据不足或接近，不硬判 winner。
- 禁止把低价 SKU 的价格错误挂到标题里的其他规格上。

## 当前实现策略

整体策略是：

```text
用户问题
 -> 解析/补全对比商品 ID
 -> LLM 抽取用户关心的对比维度
 -> 后端用维度 aliases 检索商品库证据
 -> 后端算法打分和选 winner
 -> 返回结构化 comparison + 自然语言答案
```

LLM 只负责语义理解，不负责事实判断。它输出：

```json
{
  "price_led": false,
  "dimensions": [
    {
      "label": "佩戴舒适度",
      "aliases": ["佩戴", "贴耳", "胀耳", "小耳", "滑"],
      "preference": "higher_is_better",
      "asked": true
    }
  ]
}
```

`price_led`（顶层）表示这次对比是不是主要在比价格；`asked` 表示该维度是用户在问题里明确点名要比的（如“哪个更保湿”里的保湿），其余是后端主动补充的维度。LLM 被要求在价格主导时不要为了凑数补上用户没点名的质量维度。

后端会校验这些 aliases 是否能在当前商品证据中命中。**没命中的维度会被丢弃，但只在它不是 `asked` 时**——用户明确点名却没有任何商品证据的维度（如对两件 T 恤问“哪个更防水”）会保留成一个无证据标记，用来如实披露“商品库里没有关于X的明确信息”，而不是悄悄丢掉。LLM 不可用、返回坏 JSON、或没有抽到有效维度时，会 fallback 到 deterministic 动态抽取。

## 商品 ID 解析

支持几种来源：

- 前端直接传 `compare_product_ids`
- 前端通过 `client_context.compare_product_ids` 传选中商品
- 用户输入里直接包含 `p_beauty_007` 这类 ID
- 用户说“第一个和第二个”，后端从当前 session 的最近推荐商品里解析
- 用户说“这两款/这两个”，后端默认取最近推荐的前两个商品
- 用户输入商品名或完整标题，后端尝试从 catalog 匹配

如果只确认到一款或没有上下文，会返回 clarification，让用户补充要对比的商品。

## 使用的数据结构

### 请求

`ChatRequest` 里新增/使用：

```json
{
  "message": "第一个和第二个哪个更保湿？",
  "session_id": "demo-session",
  "compare_product_ids": ["p_beauty_007", "p_beauty_012"],
  "client_context": {
    "recent_product_ids": ["p_beauty_007", "p_beauty_022", "p_beauty_012"],
    "compare_product_ids": []
  }
}
```

实际优先级：

```text
compare_product_ids + client_context.compare_product_ids
 -> 文本里的 product_id
 -> ordinal reference，例如 第一个/第二个
 -> 商品名匹配
 -> 最近推荐上下文
```

### 响应

`ChatResponse.comparison` 返回：

```json
{
  "focus": ["保湿效果"],
  "rows": [
    {
      "dimension": "价格与SKU",
      "winner_product_id": "p_beauty_007",
      "values": [
        {
          "product_id": "p_beauty_007",
          "value": "89元起（15g 体验装）",
          "evidence": ["15g 体验装 89元；50g 标准装 268元"],
          "confidence": "high"
        }
      ],
      "verdict": "..."
    }
  ],
  "winner_product_id": "p_beauty_007",
  "recommendation": "...",
  "summary": "..."
}
```

### 商品事实来源

对比功能使用这些字段：

- `title`
- `brand`
- `category`
- `sub_category`
- `skus`
- `rag_knowledge.marketing_description`
- `rag_knowledge.official_faq`
- `rag_knowledge.user_reviews`

其中价格和规格只从 `skus` 走：

```json
{
  "skus": [
    {
      "sku_id": "s_p_beauty_007_1",
      "properties": {"规格": "15g 体验装"},
      "price": 89.0
    },
    {
      "sku_id": "s_p_beauty_007_2",
      "properties": {"规格": "50g 标准装"},
      "price": 268.0
    }
  ]
}
```

## 评分逻辑

一次对比最多比 3 个商品。每次先出三行固定的结构化行——`基础定位`、`价格与SKU`、`规格明细`——再加若干 evidence row（聚焦维度最多 4 个）。

evidence row 的判定对每个商品读取：

```text
商品标题
SKU 文本
商品描述 marketing_description
官方问答 official_faq
用户评价 user_reviews
```

每个维度由 LLM evidence judge（`_llm_judge`）给出本行赢家、理由和置信度，但**带 grounding 闸门**：判词引用的片段必须确实出现在该商品上面那几段文本里（`normalize(quote) in grounding[pid]`），否则把高 / 中置信度降到 low，不让它凭空给商品安属性。LLM 不可用时退回基于 aliases 命中 + 通用正负语气的确定性评分。每行还要赢得够明显才算数：本行最高分必须 `> 0` 且比第二名高 `>= 2`，否则这一行不出 winner。

价格 row 是特殊规则：

- 使用 `lowest_price_sku` 或用户指定规格的 `selected_price_sku`
- winner 由结构化 SKU 价格决定
- 推荐文案必须明确具体 SKU，例如 `薇诺娜 15g 体验装（89元）`
- 不允许把 `15g` 的价格说成 `50g` 的价格

### winner 跨维度加权（`_recommend`）

最终 `winner_product_id` 不是看单行，而是把各行的“本行赢家”按权重累加，分高者胜：

- 用户明确点名的维度（`asked`）权重 `5.0`——它应当主导结论，即使后端又补了相关维度。
- 后端主动补充的质量维度权重 `2.0`（价格主导模式下压成 `0.0`，见下）。
- 基础定位 / 规格明细行权重 `0.0`（只展示，不参与定胜负）。
- 价格行：价格主导时 `1.5`，否则 `0.0`。

如果累加后最高分仍 `<= 0`（证据不足或太接近），不硬判 winner，返回 `None` 并说明各有侧重。

### 价格主导对比（price-led）

当用户主要在比价格时进入价格主导模式，触发来自两路 LLM 信号之一：意图解析器的 `prefer_low_price`（`_price_is_priority`），或维度抽取器输出的 `price_led` / 一个 `价格` 维度。此时：

- 价格维度被提到最前，确保它能挺过“最多 4 个维度”的截断（否则四个质量维度会把它挤掉，便宜者反而判不出来）。
- 价格行拿 `1.5` 分，而所有质量维度被压成 `0.0`——用户问的是价格、不是这些维度，所以便宜的那款胜出。
- `PRICE_LED_SPEC` 不带检索词，价格由结构化字段直接判定。

### 如实披露无证据维度

用户明确问、但商品库里没有任何证据的维度（如对两件 T 恤问“哪个更防水”），不会被悄悄丢掉，而是拼一句“商品库里没有关于{X}的明确信息。”附在结论前，避免拿别的维度硬判。

## 已修复的价格口径问题

问题复现：

1. 用户先问：

```text
推荐一个适合敏感肌的保湿护肤品，cheaper is better
```

2. 再问：

```text
第一个和第二个哪个更便宜？
```

之前的问题：

- `p_beauty_007` 的最低价 SKU 是 `15g 体验装 89元`
- 但商品标题是 `...面霜50g`
- 旧推荐文案直接复用商品标题，容易让用户误以为“50g 正装 89 元”或“50g 比别人更便宜”

现在的修复：

- 价格比较 winner 仍可基于最低价 SKU
- 但 summary/recommendation 会写成：

```text
更推荐「薇诺娜 15g 体验装（89元）」...
```

如果用户要比较 `50g`，需要明确说：

```text
50g 的第一个和第二个哪个更便宜？
```

系统会按对应 SKU 比价。

## 黑箱评测

新增测试文件：

```text
tests/test_comparison_evaluation.py
```

它不调用内部函数，只通过 `/api/chat` 做黑箱测试，并用 fake LLM 模拟维度抽取。

运行：

```bash
python -m pytest -q tests/test_comparison_evaluation.py
```

完整后端测试：

```bash
python -m pytest -q
```

注意：Milvus Lite 测试在 sandbox 里可能因为 Unix socket bind 权限失败。需要在非 sandbox 环境重跑完整测试。

## 可复制调试文案

### 1. 敏感肌保湿推荐 + 保湿对比

先发：

```text
推荐一个适合敏感肌的保湿护肤品，cheaper is better
```

预期前三个：

```text
1. p_beauty_007 薇诺娜舒敏保湿特护霜
2. p_beauty_022 薇诺娜极润保湿面膜
3. p_beauty_012 理肤泉特安舒缓修复霜
```

再发：

```text
第一个和第二个哪个更保湿？
```

预期：

```text
focus: 保湿效果
winner: p_beauty_007
```

### 2. 敏感肌保湿推荐 + 价格对比

先发：

```text
推荐一个适合敏感肌的保湿护肤品，cheaper is better
```

再发：

```text
第一个和第二个哪个更便宜？
```

预期：

```text
focus: 价格 / 便宜
winner: p_beauty_007
recommendation: 薇诺娜 15g 体验装（89元）
```

必须看到 SKU 明细：

```text
15g 体验装 89元；50g 标准装 268元
```

### 3. 面霜水润不拔干

先发：

```text
推荐一个适合敏感肌的保湿护肤品，cheaper is better
```

再发：

```text
第一个和第三个哪个上脸更水润不拔干？
```

预期：

```text
focus: 上脸水润不拔干
winner: p_beauty_012
```

### 4. 跑鞋久穿舒适

先发：

```text
推荐两双适合日常训练的缓震跑鞋
```

预期前三个：

```text
1. p_clothes_009 HOKA Clifton 9
2. p_clothes_007 Nike Pegasus 41
3. p_clothes_010 特步 160X 6.0 PRO
```

再发：

```text
第一个和第二个穿一天哪个不累？
```

预期：

```text
focus: 久穿舒适度
winner: null / 不硬判
```

说明：两双都有强缓震和舒适证据，证据接近时不硬判。

### 5. 耳机通勤降噪和佩戴

先发：

```text
推荐两款适合通勤地铁的降噪蓝牙耳机
```

预期：

```text
1. p_digital_007 华为 FreeBuds Pro 5
2. p_digital_018 Apple AirPods Pro 3
```

再发：

```text
第一个和第二个哪个更安静，戴久了也舒服？
```

预期：

```text
focus: 降噪安静度 / 佩戴舒适度
winner: p_digital_007
```

### 6. 无糖气泡饮料

先发：

```text
推荐两款无糖气泡饮料
```

预期前三个：

```text
1. p_food_004 元气森林白桃味气泡水
2. p_food_024 元气森林白葡萄味苏打气泡水
3. p_food_015 可口可乐零度
```

再发：

```text
第一个和第二个哪个糖分更低、气泡口感更好？
```

预期：

```text
focus: 糖分含量 / 气泡口感
winner: p_food_004
```

## 注意事项

真实 LLM 的 `focus` 标签可能有轻微变化，例如：

```text
保湿效果 / 保湿能力
佩戴舒适度 / 久戴舒适度
降噪安静度 / 降噪效果
```

调试时应重点看：

- 是否找对商品 ID
- 是否抽到语义等价维度
- 价格/SKU 是否照抄结构化字段
- winner 是否有足够证据
- 证据不足时是否没有硬判
