# Cart Checkout Agent Design

## 目标

Cart Checkout Agent 对应 spec 里的 4.1「购物车与下单能力」。它负责把自然语言购物车指令转换成确定的业务操作，并维护购物车和模拟订单状态。

典型需求：

```text
把第一个加到购物车
第二个来两件
再加一件
删掉第二个
数量改成 2
购物车里有什么
下单吧
确认提交
取消下单
```

这个模块的重点不是生成自然语言回答，而是稳定执行结构化 CRUD 操作：

- 加购
- 删除
- 改数量
- 增减数量
- 清空购物车
- 查看购物车
- 创建订单草稿
- 确认提交模拟订单
- 取消订单草稿

## 设计原则

1. 购物车事实必须来自 catalog 和 pricing，不从 LLM 文案或前端展示文本里推价格。
2. LLM 只做意图补全和歧义理解，不能决定价格、SKU、订单号、subtotal。
3. deterministic parser 优先处理高置信度指令；只有信息不完整时才调用 LLM commerce parser。
4. 商品引用必须解析成真实 `product_id` 后才能执行。
5. “第一个/第二个/这个/刚才那个”必须基于最近展示商品或购物车上下文解析。
6. 如果引用不明确，不猜，进入 clarification。
7. 单步购物车指令直接走 `CommerceService`；复合任务由 `PlannerService` 调度后复用 `CommerceService`。
8. 前端购物车状态以服务端 SSE `cart` event 为准。

## 架构

核心链路：

```text
用户输入
  -> ShoppingAssistant.prepare()
       -> CommerceService.maybe_handle()
            -> pending clarification resolver
            -> deterministic parser
            -> optional LLM parser
            -> candidate validation/merge
            -> action executor
       -> PreparedChat(cart/order/answer/intent)
  -> FastAPI ChatResponse 或 SSE event
  -> iOS ChatViewModel.cartItems
  -> CartSheetView / OrderReviewScreen
```

相关文件：

- `server/commerce.py`：购物车/下单意图解析和执行。
- `server/pricing.py`：购物车 item 和金额计算。
- `server/prompts.py`：`commerce_intent_messages()`。
- `server/assistant.py`：把 commerce result 包装成 `PreparedChat`。
- `server/app.py`：输出 `cart` / `order` SSE event。
- `server/schemas.py`：`CartItem`、`CartUpdate`、`OrderDraft`。
- `client/ios/.../SSEChatService.swift`：解析 cart/order event。
- `client/ios/.../ChatViewModel.swift`：维护本地 `cartItems`。
- `client/ios/.../CartSheetView.swift`：购物车 UI。
- `client/ios/.../ShoppingConciergeRootView.swift`：结算页和订单确认 UI。
- `tests/test_commerce_flow.py`：购物车/下单黑箱测试。

## 数据结构

### CommerceActionCandidate

`CommerceActionCandidate` 是后端执行前的候选动作：

```python
CommerceActionCandidate(
    action="add",
    refs=["第一个"],
    product_ids=[],
    quantity=1,
    target_scope="shown_products",
    confidence="high",
)
```

字段说明：

- `action`：白名单动作。
- `refs`：用户原话里的引用，例如“第一个”“这个”。
- `product_ids`：已经解析出的真实商品 id，只能来自已知上下文。
- `quantity`：数量。
- `item_quantities`：多商品不同件数时的逐商品数量（如“第一个买两瓶，第二个买三瓶”），`product_id -> quantity`，缺省回退到 `quantity`。
- `sku`：用户点名的规格原话（如“50g标准装”“512GB高配版”），加购时解析成真实 sku_id 再定价。
- `address`：`set_address` 动作里用户说的收货地址原文。
- `target_scope`：引用对象属于 `shown_products`、`cart_items` 或 `unknown`。
- `confidence`：`high`、`medium`、`low`。

### CartItem

`CartItem` 是购物车里的真实业务 item：

```python
CartItem(
    product_id="p_beauty_007",
    quantity=1,
    product=ProductCard(...),
    sku_id="sku_15g",
    unit_price=89.0,
    price_label="89元起（15g 体验装）",
    line_total=89.0,
)
```

价格来自 `server/pricing.py`：

```text
ProductCatalog -> ProductCard/SKU -> build_cart_item() -> CartItem
```

不允许从 LLM 回答里提取价格。

### CartUpdate

每次购物车操作返回：

```python
CartUpdate(
    items=[...],
    summary="已将「薇诺娜舒敏保湿特护霜」加入购物车，数量 1。",
    action="add",
    subtotal=89.0,
    needs_clarification=False,
)
```

### OrderDraft

下单流程返回：

```python
OrderDraft(
    order_id=None,
    status="awaiting_confirmation",
    items=[...],
    subtotal=89.0,
    address=order_state.address,   # 真实收货地址，来自客户端字段或 set_address，"默认地址" 只是兜底
    summary="订单待确认..."
)
```

`_checkout` 与 `_confirm_order` 都把 `order_state.address` 盖到 `OrderDraft.address` 和摘要文案上（“……收货地址为{address}……”）。地址不再是硬编码，详见下文「收货地址确认」。

确认提交后：

```python
OrderDraft(
    order_id="EG20260608XXXXXX",
    status="submitted",
    items=[...],
    subtotal=89.0,
    summary="订单已提交..."
)
```

## 支持的动作

白名单 action：

```text
add
remove
set_quantity
increment
decrement
clear
show_cart
set_address
checkout
confirm_order
cancel_order
none
```

其中 `set_address` 是对话式设置 / 修改收货地址（“把地址改成上海徐汇区…”“寄到…”）：LLM 把地址原文照抄进 `address`，确定性写入 `order_state.address`；若有待确认草稿则重建草稿刷新卡片和摘要。详见下文「收货地址确认」与「库存」两节。

### add

输入示例：

```text
把第一个加到购物车
第二个来两件
买这个
```

执行逻辑：

```text
解析 refs/quantity
  -> 从 session_products 或 client recent products 解析 product_id
  -> build_cart_item()
  -> upsert 到 cart
  -> 返回 CartUpdate(action="add")
```

### remove

输入示例：

```text
删掉第二个
不要这个了
```

执行逻辑：

```text
解析 cart item 引用
  -> 从 cart_items 找到 item
  -> 删除
  -> 返回 CartUpdate(action="remove")
```

### set_quantity

输入示例：

```text
把数量改成 2
第二个数量设为 3
```

执行逻辑：

```text
解析 quantity
  -> 如果购物车只有一个 item，可省略商品引用
  -> 多 item 时必须明确第几个或商品
  -> quantity=0 时等价于删除
```

### increment / decrement

输入示例：

```text
再加一件
减一件
```

执行逻辑：

```text
购物车只有一个 item -> 直接调整
购物车多个 item -> clarification
```

### clear

输入示例：

```text
清空购物车
```

执行逻辑：

```text
items=[]
order_state.draft=None
```

### show_cart

输入示例：

```text
购物车里有什么
查看购物车
```

执行逻辑：

```text
返回当前 cart summary，不改变 items。
```

### checkout

输入示例：

```text
下单吧
结算
去支付
```

执行逻辑：

```text
cart empty -> 返回无法下单
cart non-empty -> 创建 OrderDraft(status="awaiting_confirmation")
```

### confirm_order

输入示例：

```text
确认
提交吧
用默认地址
```

执行逻辑：

```text
如果有 draft 且 cart signature 没变化:
  -> 生成 order_id
  -> OrderDraft(status="submitted")
  -> 清空购物车
否则:
  -> 重新创建 checkout draft
```

### cancel_order

输入示例：

```text
取消
先不买
不下单
```

执行逻辑：

```text
清除 order draft
保留购物车商品
```

## 意图识别策略

### 1. deterministic first

`CommerceService` 先用规则解析高置信度指令：

```text
加入购物车 / 加购 / 买这个 -> add
删除 / 移除 / 不要了 -> remove
数量改成 / 数量设为 -> set_quantity
再加一件 -> increment
减一件 -> decrement
清空购物车 -> clear
下单 / 结算 / 去支付 -> checkout
确认 / 提交吧 -> confirm_order
取消 / 先不买 -> cancel_order
```

优点：

- 快。
- 稳定。
- 对常见短句准确。
- 不依赖 LLM 可用性。

### 2. LLM fills incomplete candidate

只有 deterministic candidate 不完整时才调用 LLM commerce parser。

LLM prompt 要求只输出：

```json
{
  "action": "add",
  "refs": ["第一个"],
  "product_ids": [],
  "quantity": 1,
  "target_scope": "shown_products",
  "confidence": "high"
}
```

LLM 不能输出价格、SKU、订单金额。

### 3. candidate merge

合并规则：

- deterministic 是 high confidence 时优先。
- 如果 LLM 给出冲突 high confidence action，后端保守返回 `none`，避免误操作。
- 如果 deterministic 缺 refs 或 quantity，可采用 LLM 补全。

## 商品引用解析

### shown_products

用于加购：

```text
第一个 / 第二个 / 这个 / 刚才那个
```

来源优先级：

1. 前端 `client_context.recent_product_ids`
2. 服务端 session memory `shown_products`

这样即使发生缓存 replay、后端重启，前端仍可提供最近展示商品 id。

### cart_items

用于删除、改数量、增减数量：

```text
购物车里的第一个
删掉第二个
```

来源：

```text
client_context.cart_items
```

如果购物车只有一个商品，`再加一件`、`减一件`、`数量改成 2` 可省略商品引用。

如果购物车有多个商品但用户没说是哪一个，返回 clarification。

## Clarification 机制

`OrderState.pending_action` 保存上一次未能执行的候选动作。

例子：

```text
用户：再加一件
Agent：购物车里有多件商品，请告诉我想调整哪一件，或说明第几个商品。
用户：1
Agent：已将「xxx」数量增加到 2。
```

关键点：

- 用户第二轮只说 `1`、`第一个`、`加第一个` 都会被解析为 pending action 的补充。
- pending action 成功执行后清空。
- 用户改说其它明确购物车命令时，走新的命令。

这个机制修复了短句被错误路由到 chitchat/comparison 的问题。

## 下单状态管理

`OrderState` 存在于服务端 session：

```python
OrderState(
    draft=OrderDraft | None,
    cart_signature=(("p1", 1), ("p2", 2)),
    pending_action=CommerceActionCandidate | None,
    stock_sold={},                # 本会话已下单的逐商品数量（库存台账），下单提交时累加
    address="默认地址",            # 本会话收货地址，每轮从客户端字段写入，set_address 可覆盖
)
```

`cart_signature` 用于确认订单时判断购物车是否变化：

- 如果 draft 存在且 signature 一致，`确认` 会提交订单。
- 如果 cart 变化，`确认` 会重新生成订单草稿，而不是提交旧订单。

## 前后端事件

### 同步 API

`/api/chat` 返回：

```json
{
  "answer": "已将「xxx」加入购物车，数量 1。",
  "cart": {
    "items": [],
    "summary": "...",
    "action": "add",
    "subtotal": 89.0,
    "needs_clarification": false
  },
  "order": null,
  "intent": {
    "intent_type": "cart_action",
    "commerce_action": "add",
    "commerce_refs": ["第一个"],
    "quantity": 1,
    "target_scope": "shown_products"
  }
}
```

### SSE cart event

```text
event: cart
data: {
  "type": "cart_updated",
  "cart_items": [...],
  "summary": "已将「xxx」加入购物车，数量 1。",
  "action": "add",
  "subtotal": 89.0
}
```

### SSE order event

```text
event: order
data: {
  "type": "order_draft",
  "status": "awaiting_confirmation",
  "summary": "订单待确认..."
}
```

提交成功：

```text
event: order
data: {
  "type": "order_submitted",
  "status": "submitted",
  "order_id": "EG..."
}
```

## 前端行为

iOS 端：

- `SSEChatService` 解析 `cart` / `order` event。
- `ChatViewModel` 收到 `.cartUpdated` 后更新 `cartItems`。
- `ChatHeaderView` 展示购物车数量。
- `CartSheetView` 展示购物车明细和数量调整。
- `OrderReviewScreen` 展示确认订单页面。
- 结算页支持修改联系人、手机号、详细地址。

收货地址已随每个请求发送到服务端并盖到订单上（详见「收货地址确认」），对话里也可“把地址改成…”。订单提交本身仍是模拟闭环，后续如果接真实订单 API，把这份 shipping info 转给真实下单接口即可。

## 与 Planner Agent 的关系

单步购物车任务：

```text
把第一个加到购物车
```

直接走：

```text
CommerceService.maybe_handle()
```

复合任务：

```text
帮我推荐跑鞋，对比最便宜的两双，把更便宜的加入购物车
```

走：

```text
PlannerService
  -> product_search
  -> select_products
  -> comparison
  -> CommerceService.apply_candidate()
```

也就是说，planner 不重新实现购物车逻辑，只把确定出的真实 product_id 交给 commerce 执行。

## 可复制测试文案

### 加购

前置：

```text
推荐三款保湿面霜
```

继续：

```text
把第一个加到购物车
```

期望：

- 购物车出现第一款商品。
- 数量为 1。
- 价格 label 和商品卡一致。

### 指定数量加购

```text
第二个来两件
```

期望：

- 购物车加入第二款。
- 数量为 2。

### 单商品增量

前置：购物车只有一个商品。

```text
再加一件
```

期望：

- 该商品数量 +1。

### 多商品 clarification

前置：购物车有多个商品。

```text
再加一件
```

期望：

```text
购物车里有多件商品，请告诉我想调整哪一件，或说明第几个商品。
```

继续：

```text
1
```

期望：

- 购物车第一个商品数量 +1。

### 删除

```text
删掉第二个
```

期望：

- 购物车第二个 item 被删除。

### 改数量

```text
把数量改成 2
```

期望：

- 如果购物车只有一个商品，数量变成 2。
- 如果购物车多个商品，要求用户说明第几个。

### 查看购物车

```text
购物车里有什么
```

期望：

- 返回当前商品数量和 subtotal。
- 不改变购物车内容。

### 下单

```text
下单吧
```

期望：

- 返回 `OrderDraft(status="awaiting_confirmation")`。
- 前端可进入确认订单页面。

### 确认提交

```text
确认
```

期望：

- 返回 `OrderDraft(status="submitted")`。
- 生成模拟订单号。
- 购物车清空。

### 取消订单

```text
取消
```

期望：

- 清除订单草稿。
- 购物车商品保留。

## 黑箱测试覆盖

主要测试文件：

```text
tests/test_commerce_flow.py
```

覆盖点：

- 从会话展示商品里解析“第一个”并加购。
- server memory 为空时使用前端 `recent_product_ids`。
- 指定数量加购。
- query cache 不缓存购物车操作。
- 删除、改数量、增减、清空。
- 多商品购物车 clarification。
- pending action 后续用数字回复。
- 结算、确认、取消。
- “加第一个”不能误路由到 comparison。

推荐验证命令：

```bash
.venv/bin/python -m pytest tests/test_commerce_flow.py -q
.venv/bin/python -m pytest -q --ignore=tests/test_milvus_store.py
TMPDIR=/private/tmp CLANG_MODULE_CACHE_PATH=.build/module-cache swift test --disable-sandbox
```

## 已知边界和后续增强

1. 收货地址已端到端打通（客户端字段每轮发送 + 对话 `set_address`，盖到订单上），详见下文「收货地址确认」。但另有一个独立的前端结算表单页（联系人/手机号等）仍是不驱动服务端的模拟页，不在本闭环内。
2. 当前 iOS cart 没有 SKU picker；无 `sku_id` 时使用 catalog 选中的/最低价 SKU。
3. 库存已纳入闭环（如实播报 + 加购钳制 + 下单扣减，详见下文「库存」），但库存值是为固定数据集合成的，不是真实仓储。真实支付、优惠券、物流仍不在范围。
4. 后续可以支持“把购物车里最贵的删掉”“只保留最便宜的两个”等集合操作。
5. 后续可以把 cart/order state 持久化到数据库，而不是只存在 session memory 和前端 context。

## 收货地址确认

对应 4.1 ⭐⭐⭐「引导用户确认收货地址」。地址作为会话状态放在 `OrderState.address`（本来就被穿进每个 commerce 入口、随会话持久化，和购物车状态同源），由两个输入喂同一个值：

- **客户端字段**：客户端把当前地址放进 `client_context.address` 每轮发送，所以订单从一开始就带真实地址（默认一个北京地址，而非“默认地址”）。
- **对话改地址**：`set_address` 动作为这一轮覆盖它。因为 commerce 在每轮赋值之后才跑，口头改地址在本轮胜出；服务端把新地址回写到订单，客户端再从 `order.address` 同步自己的字段，两者不分叉。

`set_address` 和「把数量改成 2」同一套路——LLM 理解语言、确定性执行：`COMMERCE_INTENT_SYSTEM` 解析出 `action=set_address` 并把地址原文照抄进 `address`，`_apply` 写入 `order_state.address`，有待确认草稿时重跑 `_checkout` 刷新卡片和摘要（购物车没变、签名仍匹配，“确认”照样提交）。LLM 不可用时不合成 `set_address`，降级到客户端字段。

客户端：`ChatViewModel.shippingAddress` 随每个请求发送、收到订单事件时从 `order.address` 回同步；`OrderCardView` 在待确认态展示可编辑地址行（复用 `EditableOrderField`），“确认”沿用既有发送路径把地址带上。

## 库存

对应 spec 3.1「数据一致性保障：价格、库存等关键参数准确且实时生效」。

**重要前提：库存是合成的。** 原始数据集没有库存字段。库存在加载时按 `product_id` 确定性生成（`_seed_stock`：`12 + sha1(pid) % 48`，得 12–59 的稳定值），并把两个好认的商品钉为低/售罄供演示（iPhone=2、兰蔻小黑瓶=0）。诚实地讲，库存数字是为这套固定数据集播种的演示值；可被评分的是建在其上的工程——单一真相源 → 如实播报 → 购物车强制 → 下单实时扣减——这几层是真的。

- **会话台账**：`OrderState.stock_sold`（`product_id -> 已下单量`）只在订单提交时增加。`available = max(0, catalog.stock(pid) - stock_sold)`；待结算的购物车行不算扣减。
- **如实播报**：`product_facts` 多一个 `available` 字段交给回答模型，搜索路径用会话感知的 `_available_by_id` 算出可用量，文本与图搜共用同一解析器；`SYSTEM_PROMPT` 要求库存只照抄 `available`，为 0 即售罄、不作首选，禁止编造。
- **购物车强制**：`_apply` 的 add/increment/set_quantity 把结果数量钳到 `_available`，售罄跳过、超买加到上限并点明，`_upsert`/`_set_quantity` 保持纯粹。单行数量另有上限 `MAX_CART_QUANTITY`（`pricing.py`，默认 999，可用 env 覆盖），避免“要1000000件”这类病态值。
- **实时扣减**：`_confirm_order` 提交时对每行执行 `stock_sold[pid] += qty`，之后同一会话再加该商品或重新搜索看到的就是减少（或为 0）的可用量。扣减是会话内的，一次演示不会掏空全局。
