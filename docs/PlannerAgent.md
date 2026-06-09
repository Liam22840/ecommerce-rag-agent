# Planner Agent Design

## 目标

Planner Agent 用来支持复合购物任务，例如：

> 帮我推荐跑鞋，对比最便宜的两双哪个更便宜，把更便宜的加入购物车。

在没有 planner 之前，系统主要按单个 `intent_type` 路由：搜索、对比、购物车、结算只能选择一个主路径。这会导致复合句被某个模块提前抢走，例如购物车规则看到“加入购物车”后直接执行，跳过前面的推荐和对比。Planner 的目标是把一句复合需求拆成一串可验证、可执行的步骤，再调用现有模块完成每一步。

## 设计原则

1. Planner 只负责拆任务和调度，不负责判断商品事实。
2. 商品价格、SKU、标题、类目、对比 winner、购物车金额都必须来自现有 catalog/comparison/commerce 模块。
3. LLM 可用于理解复合意图，但输出只能是白名单 schema，后端必须验证后再执行。
4. 没有 LLM 时必须可降级：deterministic fallback 支持常见复合链路。
5. 单步任务不能被 planner 误接管，例如“把第一个加到购物车”仍走购物车模块。
6. 前端只展示后端 plan 状态，不在客户端推理业务逻辑。

## 架构

整体链路：

```text
用户输入
  -> PlannerService.plan()
       -> LLM planner JSON 或 deterministic fallback
       -> PlannedTask / PlannedStep
  -> ShoppingAssistant._prepare_planned_task()
       -> product_search step 调用现有 IntentParser + Retriever
       -> select_products step 在真实 ProductCard 上排序/截取
       -> comparison step 调用 ComparisonService
       -> cart_action step 调用 CommerceService
       -> checkout step 调用 CommerceService
  -> PreparedChat(plan, products, comparison, cart, order, answer)
  -> FastAPI ChatResponse 或 SSE event
  -> iOS ChatViewModel timeline
  -> PlanStatusView 灰色步骤卡片
```

新增/修改的主要文件：

- `server/planner.py`：PlannerService、PlannedTask、PlannedStep、deterministic fallback。
- `server/prompts.py`：`planner_messages()` 和 planner system prompt。
- `server/assistant.py`：在普通 commerce/search/comparison 路由前执行复合 planner，并调度每一步。
- `server/schemas.py`：新增 `PlanStep`、`ExecutionPlan`，`ChatResponse.plan`。
- `server/app.py`：同步响应返回 `plan`，流式接口新增 `event: plan`。
- `server/commerce.py`：暴露 `apply_candidate()`，让 planner 能复用购物车执行逻辑。
- `client/ios/.../ChatModels.swift`：新增 `PlanStep`、`.plan` timeline/event。
- `client/ios/.../SSEChatService.swift`：解析 `event: plan`。
- `client/ios/.../ChatViewModel.swift`：收到 `.plan` 后插入 timeline。
- `client/ios/.../PlanStatusView.swift`：前端灰色 plan 步骤展示。
- `tests/test_planner_flow.py`：后端黑箱测试。
- Swift tests：SSE plan parser 和 ViewModel timeline 测试。

## Planner Schema

LLM planner 只能输出 JSON 或 `null`。单步需求输出 `null`，复合需求输出：

```json
{
  "steps": [
    {
      "action": "product_search",
      "title": "推荐跑鞋",
      "query": "推荐跑鞋"
    },
    {
      "action": "select_products",
      "title": "筛选最低价",
      "criteria": "price_asc",
      "count": 2
    },
    {
      "action": "comparison",
      "title": "对比候选商品",
      "criteria": "price_asc"
    },
    {
      "action": "cart_action",
      "title": "加入购物车",
      "target": "comparison_winner",
      "quantity": 1
    }
  ]
}
```

白名单 action：

- `product_search`：搜索/推荐商品。
- `select_products`：从上一步真实商品里选择候选。
- `comparison`：对已选商品做对比。
- `cart_action`：把已确定商品加入购物车。
- `checkout`：创建订单草稿。
- `ask_clarification`：信息不足时请求用户补充。

白名单 selection criteria：

- `price_asc`
- `price_desc`
- `rating_desc`
- `relevance`

白名单 cart target：

- `selected_products`
- `comparison_winner`
- `previous_step`

## 执行方法

### 1. 判断是否为复合任务

现在先由专注的 LLM 路由分类器（`classify_route`）判定这一轮是不是 `plan`。判定为 `plan` 时，assistant 用 `force=True` 调 planner，**绕过** `looks_like_planned_task`。`looks_like_planned_task(message)` 只在 LLM 不可用、走关键词兜底路由时作为前置判断，它要求：

- 句子里有连接关系或分句符，例如“并且 / 然后 / 再 / 同时 / 顺便 / 逗号 / 句号”。
- 至少包含两个可执行动作，例如搜索 + 加购、搜索 + 对比、对比 + 加购。

不论哪条路径，单步需求都不会被 planner 抢走（`_valid_plan` 要求至少两个动作，单步指代加购仍走购物车模块）。

### 2. 生成计划

优先使用 LLM planner：

```text
message + categories + sub_categories + brands + session_products + cart_items
  -> planner_messages()
  -> JSON plan
  -> schema validation
```

如果 LLM 不可用、返回非法 JSON、或输出不可执行 action，则 fallback 到 deterministic planner。

### 3. 执行搜索

`product_search` 不直接做检索，而是复用：

```text
IntentParser.parse()
ProductRetriever.retrieve()
ShoppingAssistant._prepare_search()
```

因此类目、品牌、预算、属性、价格排序等仍沿用现有搜索能力。如果这一步的搜索被解析成 chitchat（想要的商品不在售，如“手表”），整个计划会中止：该步标记 failed、回一句“本店暂不提供这件商品。”，而不是继续把最近邻商品硬塞进购物车。

### 4. 执行选择

`select_products` 只在真实 `ProductCard` 上操作：

- `price_asc` 按 `ProductCard.price` 升序。
- `price_desc` 按 `ProductCard.price` 降序。
- `rating_desc` / `relevance` 保留搜索排序——评分/相关度的排序是在 search 步骤里通过 `filters.sort_by` 完成的，select 步骤只再做价格排序。

这里不会让 LLM 写商品 id，也不会让 LLM 自己判断价格。

### 5. 执行对比

`comparison` 使用上一步选出的真实 product ids 调用：

```text
ComparisonService.build()
```

如果对比标准是价格低优先，会设置：

```python
filters.prefer_low_price = True
filters.sort_by = "price_asc"
```

最终 winner 来自 comparison 模块。

### 6. 执行购物车

`cart_action` 使用真实 product id 构造 `CommerceActionCandidate`，再调用：

```text
CommerceService.apply_candidate()
```

这样购物车 item、SKU、unit price、line total、subtotal 全部来自统一 commerce/pricing 逻辑。

cart 步骤还会把 search 步骤解析出的 `requested_specs`（用户点名的规格，如“512GB高配版”）透传进 `_apply_cart_targets`，拼成 `candidate.sku`，让 planner 加购的那一行按该 SKU 定价，而不是默认回退到最便宜 SKU——和直接加购路径一致。

### 7. 返回和展示计划

同步 API：

```json
{
  "plan": {
    "steps": [
      {"step_id": "step-1", "title": "推荐跑鞋", "action": "product_search", "status": "done"},
      {"step_id": "step-2", "title": "加入购物车", "action": "cart_action", "status": "done"}
    ]
  }
}
```

流式 API：

```text
event: plan
data: {"type":"plan","steps":[...]}
```

iOS 前端展示为灰色步骤卡片：

```text
✓ 推荐跑鞋
✓ 筛选最低价
✓ 加入购物车
```

## 适配例子

### 例子 1：推荐后直接加购

输入：

```text
帮我推荐跑鞋，并把最便宜的一双加入购物车
```

计划：

```text
1. product_search: 推荐跑鞋
2. select_products: 按 price_asc 选择 1 款
3. cart_action: 加入 selected_products
```

期望：

- 返回跑鞋商品卡。
- 选中真实返回商品里价格最低的一款。
- 购物车里加入该真实 product_id。

### 例子 2：推荐、筛选、对比、加购

输入：

```text
帮我推荐跑鞋，对比最便宜的两双哪个更便宜，把更便宜的加入购物车
```

计划：

```text
1. product_search: 推荐跑鞋
2. select_products: 按 price_asc 选择 2 款
3. comparison: 对比价格
4. cart_action: 加入 comparison_winner
```

期望：

- 对比对象是搜索结果里真实最低价的两款。
- winner 来自 comparison 模块。
- 加购商品 id 等于 `comparison.winner_product_id`。

### 例子 3：搜索后下单

输入：

```text
帮我找一款蓝牙耳机，然后加入购物车并结算
```

计划：

```text
1. product_search: 推荐蓝牙耳机
2. select_products: 选择 1 款
3. cart_action: 加入购物车
4. checkout: 创建订单草稿
```

期望：

- 商品来自耳机搜索结果。
- 购物车金额和订单草稿金额一致。
- 结算页继续允许用户修改联系人、手机号、地址。

### 例子 4：单步购物车不走 planner

前一轮已经展示商品后，输入：

```text
把第一个加到购物车
```

期望：

- `plan` 为 `null`。
- 直接走 `CommerceService`。
- “第一个”按最近展示商品上下文解析。

### 例子 5：普通搜索不走 planner

输入：

```text
推荐一款适合敏感肌的保湿面霜
```

期望：

- `plan` 为 `null`。
- 走原搜索推荐链路。
- 商品事实和价格说明仍来自商品库。

## 测试策略

先写黑箱测试，再实现：

- `tests/test_planner_flow.py`
  - 复合推荐 + 最低价加购。
  - 推荐 + 选择两款 + 对比 + 加购 winner。
  - 单步购物车 follow-up 不被 planner 接管。
  - SSE 先发 `event: plan` 再发 cart。
  - Fake LLM 输出通用 planner schema，后端仍必须按真实 catalog 和 cart 执行。

前端测试：

- `SSEEventParserTests.testParsesPlannerEvent`
- `ChatViewModelFlowTests.testPlanEventAppendsPlanTimelineItem`

已验证命令：

```bash
.venv/bin/python -m pytest tests/test_planner_flow.py -q
.venv/bin/python -m pytest -q --ignore=tests/test_milvus_store.py
TMPDIR=/private/tmp CLANG_MODULE_CACHE_PATH=.build/module-cache swift test --disable-sandbox
git diff --check
.venv/bin/python -m py_compile server/planner.py server/assistant.py server/app.py server/commerce.py
```

验证结果：

- Planner 黑箱测试：5 passed。
- 后端应用层测试：357 passed。
- Swift package tests：33 executed, 1 skipped, 0 failures。
- 全量 pytest 中 `tests/test_milvus_store.py` 的 3 个测试在当前沙箱下无法绑定本地 Unix socket，属于环境限制，不是 planner 逻辑失败。

## 后续可扩展点

> 注：每一步实时 running/done 的多次 `event: plan` 流式更新已经实现（`_prepare_planned_task_updates`：循环前先发一帧，每步置 running 再发，done/failed 时再发），不再是一次性返回最终状态。

1. 支持更多 action，例如 `remove_from_cart`、`set_quantity`、`apply_coupon`。
3. 支持更细的 clarification，例如“你是想对比最便宜的两款，还是把最便宜的一款直接加入购物车？”
4. 将 planner execution trace 存入 session，方便用户问“刚才为什么选这个？”。
5. 给 planner 增加 offline evaluation set，按真实用户复合句统计路由准确率。
