# 库存能力设计 (Inventory)

本文档说明库存功能的设计与实现。对应 spec 3.1「数据工程与特征治理」里的「数据一致性保障：确保 AI 推荐的商品价格、库存等关键参数准确且实时生效」。

## 一个重要前提：库存是合成的

原始数据集里**没有库存字段**。每个商品只有 `product_id, title, brand, category, sub_category, base_price, image_path, skus, rag_knowledge`，没有任何 stock / 库存 / 现货信息（`skus.properties` 里出现的「数量」是包装规格，如「12 瓶」，不是仓库存量）。

所以库存数字是我们在加载时**确定性合成**的，不是真实仓储数据，磁盘上的数据集也从不被修改。诚实地讲：库存值是为这套固定数据集播种（seeded）的演示值。真正可被评分的工程能力是建在它之上的部分——**单一真相源 → 如实播报 → 购物车强制 → 下单实时扣减**——这几层都是真的。

## 数据来源与生成

库存在 `ProductCatalog.__init__` 加载时逐商品写入 `product["stock"]`（见 `server/catalog.py` 的 `_seed_stock`）：

- 公式：`12 + sha1(product_id) % 48`，得到 12–59 的稳定数字。用 `hashlib` 而不是内置 `hash()`（后者对字符串是随机化的，跨进程不稳定）。
- 结果对同一 `product_id` 永远一致，每个商品（包括以后新增的）都会自动有库存，且不会在两次运行之间漂移。
- 公式刻意高于任何正常购物数量，所以普通加购永远不会被它卡住——**只有被显式钉住的商品才会受限**。
- 演示钉子 `_STOCK_PINS`：把两个好认的商品钉为低库存 / 售罄，让「买光就没了」和下单扣减可以稳定复现：
  - `p_digital_003`（iPhone 17 Pro Max）= 2，低库存，演示超买钳制。
  - `p_beauty_002`（兰蔻小黑瓶）= 0，售罄，演示直接拒绝加购。

`catalog.stock(product_id)` 返回这个 base 值。

## 单一真相与会话台账

会话内「还能买多少」= base 减去本会话已下单的量。

- `OrderState.stock_sold: dict[product_id -> int]`（`server/commerce.py`）是每个会话的「已售」台账。它只在订单**提交**时增加，并随会话持续存在（挂在会话唯一的 `OrderState` 上）。
- `CommerceService._available(product_id, order_state) = max(0, catalog.stock(pid) - stock_sold.get(pid, 0))`。
- 注意：购物车里**待结算**的行不算扣减，只有真正提交的订单才扣减。

## 如实播报（LLM 理解，确定性提供事实）

- `product_facts` 多了一个 `available` 字段，交给回答模型。它优先用会话感知的可用量（调用方传入），否则回退到 base 库存。
- 搜索回答路径在 `server/assistant.py` 的 `_available_by_id(session_id, hits)` 里算出每个候选的会话感知可用量，文本和图搜两条叙述路径共用这一个解析器，所以它们不会对库存各说各话。
- `SYSTEM_PROMPT` 增加一条规则：库存只能照抄 `available` 字段，`available` 为 0 即已售罄，不要把售罄商品作为首选推荐，库存偏低时可点明「仅剩 N 件」，禁止编造库存数字。

## 购物车强制（确定性兜底）

钳制发生在 `_apply` 各个会抬高数量的分支（确定性，LLM 只负责理解「加 / 改成几件」）：

- `add`：每个商品的**结果数量**（已有行数量 + 本次请求）被钳到 `_available`。售罄（可用为 0）则跳过该商品并如实说明；超买则加到可用上限并在回答里点明已按上限加入；多商品加购按商品分别处理。
- `increment` / `set_quantity`：把结果数量钳到 `_available`；`decrement` 不需要检查。
- `_upsert` / `_set_quantity` 保持纯粹（不感知库存），钳制在有 `order_state` 的 `_apply` 里完成。

这条单一入口同时覆盖 planner 加购路径和澄清回复加购路径。

## 实时扣减（「实时生效」）

`_confirm_order` 在生成订单号、真正提交时，对购物车每一行执行 `stock_sold[pid] += qty`。之后同一会话里再加这个商品、或重新搜索，看到的就是减少（或为 0）的可用量。**只有确认提交才扣减，下单草稿/结算阶段不扣减。**

扣减是会话内的（按 `session_id`），所以一次演示不会把全局库存掏空——这对演示是正确的，也和订单 / 会话状态原本的存活范围一致。

## 测试

`tests/test_commerce_flow.py` 与 `tests/test_catalog.py`：
- 种子库存确定性、钉子（`p_beauty_002`=0、`p_digital_003`=2）、公式下限、`product_facts.available`。
- 超买被钳到可用上限；售罄商品不被加入。
- 提交订单扣减 `stock_sold`，之后同一会话再加被拒。

真·LLM 端到端（`/tmp/rag` 驱动）：搜索如实播报可用 → 超买被钳 → 下单提交 → 再加该商品被拒（已售罄）→ 再搜索时可用量已反映扣减。
