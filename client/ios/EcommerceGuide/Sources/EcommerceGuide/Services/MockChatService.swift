import Foundation

@available(iOS 17.0, macOS 13.0, *)
public struct MockChatService: ChatService {
    private let tokenDelay: UInt64
    private let fixtureName: String

    public init(tokenDelay: UInt64 = 85_000_000, fixtureName: String = "mock_products") {
        self.tokenDelay = tokenDelay
        self.fixtureName = fixtureName
    }

    public func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let tokenDelay = tokenDelay
            let fixtureName = fixtureName

            let task = Task {
                do {
                    let products = try FixtureLoader.loadProducts(named: fixtureName)
                    let response = scriptedResponse(for: request.message)

                    for token in response {
                        try Task.checkCancellation()
                        try await Task.sleep(nanoseconds: tokenDelay)
                        continuation.yield(.token(token))
                    }

                    try Task.checkCancellation()
                    continuation.yield(.plan([
                        PlanStep(stepID: "step-1", title: "检索相关商品", action: "product_search", status: "running"),
                        PlanStep(stepID: "step-2", title: "筛选并推荐", action: "select_products", status: "pending")
                    ]))

                    // Hold the running state long enough for the card skeleton to be visible offline.
                    try await Task.sleep(nanoseconds: tokenDelay * 10)

                    try Task.checkCancellation()
                    let recommendedProducts = Array(products.prefix(3))
                    continuation.yield(.products(recommendedProducts))

                    continuation.yield(.plan([
                        PlanStep(stepID: "step-1", title: "检索相关商品", action: "product_search", status: "done", summary: "已找到候选商品"),
                        PlanStep(stepID: "step-2", title: "筛选并推荐", action: "select_products", status: "done")
                    ]))

                    if recommendedProducts.count >= 2 {
                        try Task.checkCancellation()
                        try await Task.sleep(nanoseconds: tokenDelay)
                        continuation.yield(.comparison(mockComparison(Array(recommendedProducts.prefix(2)))))
                    }

                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: tokenDelay * 2)
                    let updatedCart = mergeCartItems(request.cartItems, adding: products[0])
                    continuation.yield(.cartUpdated(updatedCart, summary: "已将「\(products[0].title)」加入购物车。"))

                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: tokenDelay)
                    continuation.yield(.done(messageID: UUID().uuidString))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func scriptedResponse(for message: String) -> [String] {
        let lowercased = message.lowercased()

        if lowercased.contains("shoe") || lowercased.contains("sneaker") || lowercased.contains("鞋") {
            return [
                "我找到了几款实用的选择。 ",
                "其中运动鞋最符合你的需求， ",
                "另外也搭配了两件好用的配套商品。"
            ]
        }

        if lowercased.contains("gift") || lowercased.contains("礼物") || lowercased.contains("送礼") {
            return [
                "这里有几款适合送礼的商品。 ",
                "我优先选择了质感好、实用性强， ",
                "并且不容易选错规格的款式。"
            ]
        }

        return [
            "我根据你的需求整理了一份精选清单。 ",
            "这些商品在日常实用性、价格， ",
            "以及目录里的商品信息之间比较均衡。"
        ]
    }

    private func mergeCartItems(_ existingItems: [CartItem], adding product: Product) -> [CartItem] {
        var items = existingItems

        if let index = items.firstIndex(where: { $0.product.id == product.id }) {
            items[index].quantity += 1
        } else {
            items.append(CartItem(product: product))
        }

        return items
    }

    private func mockComparison(_ products: [Product]) -> ProductComparison {
        ProductComparison(
            products: products,
            focus: ["价格", "实用性"],
            rows: [
                ComparisonRow(
                    dimension: "价格",
                    values: products.map {
                        ComparisonValue(productID: $0.id, value: $0.formattedPrice, confidence: "high")
                    },
                    winnerProductID: products.min(by: { $0.basePrice < $1.basePrice })?.id,
                    verdict: "按 mock 商品价格展示，低价商品更适合预算优先。"
                ),
                ComparisonRow(
                    dimension: "卖点",
                    values: products.map {
                        ComparisonValue(productID: $0.id, value: $0.reason ?? $0.spec ?? "暂无卖点信息", confidence: "medium")
                    },
                    verdict: "商品卖点来自本地 mock fixtures。"
                )
            ],
            winnerProductID: products.first?.id,
            recommendation: products.first.map { "更推荐「\($0.title)」，它更贴近当前 mock 场景。" },
            summary: "我把这几款商品做了一个快速对比。"
        )
    }
}
