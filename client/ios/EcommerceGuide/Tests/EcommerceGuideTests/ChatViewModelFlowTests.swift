import XCTest
@testable import EcommerceGuide

@MainActor
final class ChatViewModelFlowTests: XCTestCase {
    func testMandarinVoiceInputSendsFinalTranscriptThroughChatService() async throws {
        let chatService = ScriptedChatService(events: [
            .token("可以，我来推荐。"),
            .done(messageID: "voice-1")
        ])
        let speechService = ScriptedSpeechRecognitionService(updates: [
            SpeechRecognitionUpdate(transcript: "推荐一款适合油皮的洗面奶", isFinal: true)
        ])
        let viewModel = ChatViewModel(
            service: chatService,
            speechRecognitionService: speechService,
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000020")!,
            timeline: []
        )

        viewModel.toggleVoiceInput()
        try await waitUntil { chatService.requests.count == 1 }
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(speechService.startCount, 1)
        XCTAssertGreaterThanOrEqual(speechService.stopCount, 1)
        XCTAssertFalse(viewModel.isListening)
        XCTAssertEqual(chatService.requests.map(\.message), ["推荐一款适合油皮的洗面奶"])
        XCTAssertEqual(viewModel.draftMessage, "")

        guard case .message(let assistantMessage)? = viewModel.timeline.last else {
            return XCTFail("Expected final assistant message")
        }
        XCTAssertEqual(assistantMessage.text, "可以，我来推荐。")
    }

    func testSendDraftMessageReducesStreamIntoTimelineAndCart() async throws {
        let product = Product.fixture(id: "JACKET-1", title: "Rain Shell")
        let service = ScriptedChatService(events: [
            .token("I found "),
            .token("a strong option."),
            .products([product]),
            .cartUpdated([CartItem(product: product, quantity: 2)], summary: "2 Rain Shells in cart"),
            .done(messageID: "assistant-1")
        ])
        let viewModel = ChatViewModel(
            service: service,
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000010")!,
            timeline: []
        )

        viewModel.draftMessage = "Need a waterproof jacket"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.requests.map(\.message), ["Need a waterproof jacket"])
        XCTAssertEqual(viewModel.draftMessage, "")
        XCTAssertEqual(viewModel.cartItems, [CartItem(product: product, quantity: 2)])
        XCTAssertNil(viewModel.errorMessage)

        guard case .message(let userMessage) = viewModel.timeline[0] else {
            return XCTFail("Expected first item to be the user's message")
        }
        XCTAssertEqual(userMessage.role, .user)
        XCTAssertEqual(userMessage.text, "Need a waterproof jacket")
        XCTAssertFalse(userMessage.isStreaming)

        guard case .message(let assistantMessage) = viewModel.timeline[1] else {
            return XCTFail("Expected second item to be the assistant message")
        }
        XCTAssertEqual(assistantMessage.role, .assistant)
        XCTAssertEqual(assistantMessage.text, "I found a strong option.")
        XCTAssertFalse(assistantMessage.isStreaming)

        guard case .products(_, let products) = viewModel.timeline[2] else {
            return XCTFail("Expected product recommendations")
        }
        XCTAssertEqual(products, [product])

        guard case .cartStatus(_, let status) = viewModel.timeline[3] else {
            return XCTFail("Expected cart status")
        }
        XCTAssertEqual(status, "2 Rain Shells in cart")
    }

    func testSendPhotoAppendsImageBubbleAndStreamsRequestWithImageData() async throws {
        let product = Product.fixture(id: "SHOE-1", title: "Trail Runner")
        let service = ScriptedChatService(events: [
            .token("这几款接近你的图片。"),
            .products([product]),
            .done(messageID: "photo-1")
        ])
        let viewModel = ChatViewModel(service: service, conversationID: UUID(), timeline: [])

        let imageData = Data([0xFF, 0xD8, 0xFF, 0xD9])
        viewModel.sendPhoto(imageData: imageData, caption: "找同款")
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.requests.last?.imageData, imageData)
        XCTAssertEqual(service.requests.last?.message, "找同款")

        guard case .message(let userMessage) = viewModel.timeline[0] else {
            return XCTFail("Expected a user photo message")
        }
        XCTAssertEqual(userMessage.role, .user)
        XCTAssertEqual(userMessage.imageData, imageData)

        guard case .products(_, let products)? = viewModel.timeline.first(where: { item in
            if case .products = item { return true }
            return false
        }) else {
            return XCTFail("Expected product results")
        }
        XCTAssertEqual(products, [product])
    }

    func testSendPhotoUsesDefaultCaptionWhenEmpty() async throws {
        let service = ScriptedChatService(events: [.done(messageID: "photo-2")])
        let viewModel = ChatViewModel(service: service, conversationID: UUID(), timeline: [])

        viewModel.sendPhoto(imageData: Data([0xFF, 0xD8]), caption: "   ")
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.requests.last?.message, "找同款")  // blank caption -> default
    }

    func testAddToCartIncrementsExistingProductAndAppendsStatus() {
        let product = Product.fixture(id: "TEE-1", title: "Cotton Tee")
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: []),
            conversationID: UUID(),
            timeline: []
        )

        viewModel.addToCart(product: product)
        viewModel.addToCart(product: product)

        XCTAssertEqual(viewModel.cartItems, [CartItem(product: product, quantity: 2)])
        XCTAssertEqual(viewModel.timeline.count, 2)

        guard case .cartStatus(_, let status) = viewModel.timeline.last else {
            return XCTFail("Expected cart status after add to cart")
        }
        XCTAssertEqual(status, "已将「Cotton Tee」加入购物车。")
    }

    func testComparisonEventAppendsComparisonTimelineItem() async throws {
        let firstProduct = Product.fixture(id: "SUN-1", title: "First Sunscreen")
        let secondProduct = Product.fixture(id: "SUN-2", title: "Second Sunscreen")
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: [
                .token("Here is the comparison."),
                .comparison(ProductComparison(
                    products: [firstProduct, secondProduct],
                    focus: ["保湿"],
                    rows: [
                        ComparisonRow(
                            dimension: "保湿",
                            values: [
                                ComparisonValue(productID: firstProduct.id, value: "Evidence A"),
                                ComparisonValue(productID: secondProduct.id, value: "Evidence B")
                            ],
                            winnerProductID: firstProduct.id,
                            verdict: "First has stronger evidence."
                        )
                    ],
                    winnerProductID: firstProduct.id,
                    recommendation: "Pick first.",
                    summary: "Compared."
                )),
                .done(messageID: "comparison-1")
            ]),
            conversationID: UUID(),
            timeline: []
        )

        viewModel.draftMessage = "Compare the first two"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        guard case .comparison(_, let comparison)? = viewModel.timeline.first(where: { item in
            if case .comparison = item { return true }
            return false
        }) else {
            return XCTFail("Expected comparison timeline item")
        }

        XCTAssertEqual(comparison.products, [firstProduct, secondProduct])
        XCTAssertEqual(comparison.winnerProductID, firstProduct.id)
    }

    func testPlanEventAppendsPlanTimelineItem() async throws {
        let pendingSteps = [
            PlanStep(
                stepID: "step-1",
                title: "推荐跑鞋",
                action: "product_search",
                status: "pending"
            ),
            PlanStep(
                stepID: "step-2",
                title: "加入购物车",
                action: "cart_action",
                status: "pending"
            )
        ]
        let runningSteps = [
            PlanStep(
                stepID: "step-1",
                title: "推荐跑鞋",
                action: "product_search",
                status: "running"
            ),
            PlanStep(
                stepID: "step-2",
                title: "加入购物车",
                action: "cart_action",
                status: "pending"
            )
        ]
        let doneSteps = [
            PlanStep(
                stepID: "step-1",
                title: "推荐跑鞋",
                action: "product_search",
                status: "done",
                summary: "找到 3 款候选商品。"
            ),
            PlanStep(
                stepID: "step-2",
                title: "加入购物车",
                action: "cart_action",
                status: "done",
                summary: "已加入购物车。"
            )
        ]
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: [
                .token("我来执行。"),
                .plan(pendingSteps),
                .plan(runningSteps),
                .plan(doneSteps),
                .done(messageID: "plan-1")
            ]),
            conversationID: UUID(),
            timeline: []
        )

        viewModel.draftMessage = "推荐跑鞋并加入购物车"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        guard case .plan(_, let timelineSteps)? = viewModel.timeline.first(where: { item in
            if case .plan = item { return true }
            return false
        }) else {
            return XCTFail("Expected plan timeline item")
        }

        XCTAssertEqual(timelineSteps, doneSteps)
        XCTAssertEqual(viewModel.timeline.filter { item in
            if case .plan = item { return true }
            return false
        }.count, 1)
    }

    func testFollowupRequestIncludesRecentProductIDs() async throws {
        let firstProduct = Product.fixture(id: "FACE-1", title: "First Cream")
        let secondProduct = Product.fixture(id: "FACE-2", title: "Second Cream")
        let service = ScriptedChatService(events: [
            .products([firstProduct, secondProduct]),
            .done(messageID: "assistant-1")
        ])
        let viewModel = ChatViewModel(service: service, conversationID: UUID(), timeline: [])

        viewModel.draftMessage = "推荐面霜"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        viewModel.draftMessage = "第一个和第二个哪个更保湿"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.requests.last?.recentProductIDs, ["FACE-1", "FACE-2"])
    }

    func testStreamCompletionWithoutDoneClearsSendingState() async throws {
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: [
                .token("Partial answer.")
            ]),
            conversationID: UUID(),
            timeline: []
        )

        viewModel.draftMessage = "Find a jacket"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertFalse(viewModel.isSending)

        guard case .message(let assistantMessage)? = viewModel.timeline.last else {
            return XCTFail("Expected assistant message after stream completion")
        }
        XCTAssertEqual(assistantMessage.text, "Partial answer.")
        XCTAssertFalse(assistantMessage.isStreaming)
    }

    func testCartStatusDoesNotReplaceExistingCartItems() async throws {
        let product = Product.fixture(id: "BAG-1", title: "Carry Bag")
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: [
                .cartStatus(summary: "Backend acknowledged cart intent."),
                .done(messageID: "cart-status")
            ]),
            conversationID: UUID(),
            timeline: []
        )
        viewModel.addToCart(product: product)

        viewModel.draftMessage = "Add it"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(viewModel.cartItems, [CartItem(product: product, quantity: 1)])
        XCTAssertTrue(viewModel.timeline.containsCartStatus("Backend acknowledged cart intent."))
    }

    func testOrderStatusDoesNotReplaceExistingCartItems() async throws {
        let product = Product.fixture(id: "BAG-2", title: "Carry Bag")
        let viewModel = ChatViewModel(
            service: ScriptedChatService(events: [
                .orderStatus(Order(status: "awaiting_confirmation", summary: "订单待确认")),
                .done(messageID: "order-status")
            ]),
            conversationID: UUID(),
            timeline: []
        )
        viewModel.addToCart(product: product)

        viewModel.draftMessage = "下单吧"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(viewModel.cartItems, [CartItem(product: product, quantity: 1)])
        XCTAssertTrue(viewModel.timeline.containsOrderStatus("订单待确认"))
    }

    func testMockChatServiceEmitsScriptedFlowWithoutNetwork() async throws {
        let service = MockChatService(tokenDelay: 0, fixtureName: "mock_products")
        let request = ChatRequest(
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000020")!,
            message: "I need sneakers"
        )

        var events: [ChatStreamEvent] = []
        for try await event in service.streamChat(for: request) {
            events.append(event)
        }

        XCTAssertTrue(events.contains(.token("我找到了几款实用的选择。 ")))

        guard case .products(let products)? = events.first(where: { event in
            if case .products = event { return true }
            return false
        }) else {
            return XCTFail("Expected mock service to emit products")
        }
        XCTAssertEqual(products.count, 3)

        guard case .comparison(let comparison)? = events.first(where: { event in
            if case .comparison = event { return true }
            return false
        }) else {
            return XCTFail("Expected mock service to emit a product comparison")
        }
        XCTAssertEqual(comparison.products.count, 2)
        XCTAssertFalse(comparison.rows.isEmpty)

        guard case .cartUpdated(let cartItems, let summary)? = events.first(where: { event in
            if case .cartUpdated = event { return true }
            return false
        }) else {
            return XCTFail("Expected mock service to emit a cart update")
        }
        XCTAssertEqual(cartItems.count, 1)
        XCTAssertTrue(summary.hasPrefix("已将「"))

        guard case .done(let messageID)? = events.last else {
            return XCTFail("Expected mock service to finish with done")
        }
        XCTAssertNotNil(messageID)
    }

    func testErrorAndRetryPreserveUserMessagesAndClearTransientError() async throws {
        let service = FailingThenSucceedingChatService(
            failure: ChatServiceError.invalidResponse,
            successEvents: [
                .token("Recovered."),
                .done(messageID: "retry-1")
            ]
        )
        let viewModel = ChatViewModel(
            service: service,
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000030")!,
            timeline: []
        )

        viewModel.draftMessage = "Find trail shoes"
        viewModel.sendDraftMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.attempts, 1)
        XCTAssertEqual(viewModel.errorMessage, "服务器返回了无效响应。")
        XCTAssertTrue(viewModel.timeline.containsError("服务器返回了无效响应。"))
        XCTAssertEqual(viewModel.userMessages.map(\.text), ["Find trail shoes"])

        viewModel.retryLastMessage()
        try await waitUntilNotSending(viewModel)

        XCTAssertEqual(service.attempts, 2)
        XCTAssertNil(viewModel.errorMessage)
        XCTAssertFalse(viewModel.timeline.containsError("服务器返回了无效响应。"))
        XCTAssertEqual(viewModel.userMessages.map(\.text), ["Find trail shoes", "Find trail shoes"])

        guard case .message(let assistantMessage)? = viewModel.timeline.last else {
            return XCTFail("Expected retry to append a successful assistant response")
        }
        XCTAssertEqual(assistantMessage.text, "Recovered.")
        XCTAssertFalse(assistantMessage.isStreaming)
    }

    private func waitUntilNotSending(
        _ viewModel: ChatViewModel,
        timeout: TimeInterval = 1,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)

        while viewModel.isSending, Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        if viewModel.isSending {
            XCTFail("Timed out waiting for ChatViewModel to finish sending", file: file, line: line)
        }
    }

    private func waitUntil(
        timeout: TimeInterval = 1,
        file: StaticString = #filePath,
        line: UInt = #line,
        condition: () -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)

        while !condition(), Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        if !condition() {
            XCTFail("Timed out waiting for condition", file: file, line: line)
        }
    }
}

private final class ScriptedChatService: ChatService, @unchecked Sendable {
    private let events: [ChatStreamEvent]
    private(set) var requests: [ChatRequest] = []

    init(events: [ChatStreamEvent]) {
        self.events = events
    }

    func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        requests.append(request)

        return AsyncThrowingStream { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }
}

private final class FailingThenSucceedingChatService: ChatService, @unchecked Sendable {
    private let failure: Error
    private let successEvents: [ChatStreamEvent]
    private(set) var attempts = 0

    init(failure: Error, successEvents: [ChatStreamEvent]) {
        self.failure = failure
        self.successEvents = successEvents
    }

    func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        attempts += 1

        return AsyncThrowingStream { continuation in
            if attempts == 1 {
                continuation.finish(throwing: failure)
                return
            }

            for event in successEvents {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }
}

private final class ScriptedSpeechRecognitionService: SpeechRecognitionService, @unchecked Sendable {
    private let updates: [SpeechRecognitionUpdate]
    private(set) var startCount = 0
    private(set) var stopCount = 0

    init(updates: [SpeechRecognitionUpdate]) {
        self.updates = updates
    }

    func startMandarinRecognition() -> AsyncThrowingStream<SpeechRecognitionUpdate, Error> {
        startCount += 1

        return AsyncThrowingStream { continuation in
            for update in updates {
                continuation.yield(update)
            }
            continuation.finish()
        }
    }

    func stopRecognition() {
        stopCount += 1
    }
}

private extension Product {
    static func fixture(
        id: String = "PRODUCT-1",
        title: String = "Fixture Product",
        brand: String = "Fixture Brand",
        category: String = "Fixture Category",
        subCategory: String = "Fixture Subcategory",
        basePrice: Decimal = Decimal(string: "42.00")!,
        imagePath: String = "images/product.jpg",
        reason: String? = nil
    ) -> Product {
        Product(
            id: id,
            title: title,
            brand: brand,
            category: category,
            subCategory: subCategory,
            basePrice: basePrice,
            imagePath: imagePath,
            reason: reason
        )
    }
}

private extension Array where Element == ChatTimelineItem {
    func containsError(_ message: String) -> Bool {
        contains {
            guard case .error(_, let itemMessage) = $0 else { return false }
            return itemMessage == message
        }
    }

    func containsCartStatus(_ message: String) -> Bool {
        contains {
            guard case .cartStatus(_, let itemMessage) = $0 else { return false }
            return itemMessage == message
        }
    }

    func containsOrderStatus(_ message: String) -> Bool {
        contains {
            guard case .orderStatus(_, let order) = $0 else { return false }
            return order.summary == message
        }
    }
}

@MainActor
private extension ChatViewModel {
    var userMessages: [ChatMessage] {
        timeline.compactMap { item in
            guard case .message(let message) = item, message.role == .user else {
                return nil
            }
            return message
        }
    }
}
