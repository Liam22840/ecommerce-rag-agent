import Foundation

@available(iOS 17.0, macOS 14.0, *)
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
                    try await Task.sleep(nanoseconds: tokenDelay * 2)
                    continuation.yield(.products(Array(products.prefix(3))))

                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: tokenDelay * 2)
                    let updatedCart = mergeCartItems(request.cartItems, adding: products[0])
                    continuation.yield(.cartUpdated(updatedCart, summary: "Cart updated with \(products[0].title)."))

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

        if lowercased.contains("shoe") || lowercased.contains("sneaker") {
            return [
                "I found a few practical picks. ",
                "The sneakers are the strongest match, ",
                "and I included two versatile add-ons that pair well with them."
            ]
        }

        if lowercased.contains("gift") {
            return [
                "Here are gift-friendly options with broad appeal. ",
                "I prioritized items that feel polished, useful, ",
                "and easy to size correctly."
            ]
        }

        return [
            "I pulled together a short list based on your request. ",
            "These balance everyday usefulness, price, ",
            "and the product details available in the catalog."
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
}
