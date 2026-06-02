import XCTest
@testable import EcommerceGuide

final class SSEEventParserTests: XCTestCase {
    private var parser: SSEEventParser!

    override func setUp() {
        super.setUp()
        parser = SSEEventParser()
    }

    override func tearDown() {
        parser = nil
        super.tearDown()
    }

    func testParsesTokenProductsCartAndDoneEvents() throws {
        let first = try parseFrame(["data: {\"type\":\"token\",\"token\":\"These boots\"}", ""])
        let second = try parseFrame(["data: {\"type\":\"token\",\"delta\":\" are durable\"}", ""])
        let products = try parseFrame([
            "event: products",
            "data: {\"items\":[{\"product_id\":\"BOOT-1\",\"title\":\"Summit Boot\",\"brand\":\"Northstar\",\"category\":\"Shoes\",\"sub_category\":\"Boots\",\"base_price\":189.00,\"price_label\":\"189元起（基础款）\",\"price_summary\":\"基础款 189元；加强款 239元\",\"image_path\":\"images/boot-1.jpg\",\"reason\":\"Good grip\"}]}",
            ""
        ])
        let cart = try parseFrame([
            "data: {\"type\":\"cart_updated\",\"summary\":\"1 item in cart\",\"cart_items\":[{\"product\":{\"product_id\":\"BOOT-1\",\"title\":\"Summit Boot\",\"brand\":\"Northstar\",\"category\":\"Shoes\",\"sub_category\":\"Boots\",\"base_price\":189.00,\"image_path\":\"images/boot-1.jpg\"},\"quantity\":1}]}",
            ""
        ])
        let done = try parseFrame(["data: {\"type\":\"done\",\"message_id\":\"msg-123\"}", ""])

        XCTAssertEqual(first, .token("These boots"))
        XCTAssertEqual(second, .token(" are durable"))
        XCTAssertEqual(products, .products([
            Product(
                id: "BOOT-1",
                title: "Summit Boot",
                brand: "Northstar",
                category: "Shoes",
                subCategory: "Boots",
                basePrice: Decimal(string: "189.00")!,
                priceLabel: "189元起（基础款）",
                priceSummary: "基础款 189元；加强款 239元",
                imagePath: "images/boot-1.jpg",
                reason: "Good grip"
            )
        ]))
        XCTAssertEqual(cart, .cartUpdated([
            CartItem(
                product: Product(
                    id: "BOOT-1",
                    title: "Summit Boot",
                    brand: "Northstar",
                    category: "Shoes",
                    subCategory: "Boots",
                    basePrice: Decimal(string: "189.00")!,
                    imagePath: "images/boot-1.jpg"
                ),
                quantity: 1
            )
        ], summary: "1 item in cart"))
        XCTAssertEqual(done, .done(messageID: "msg-123"))
    }

    func testBuffersPartialSSEFramesUntilBlankLineTerminator() throws {
        XCTAssertNil(try parser.consume(line: "data: {\"type\":\"token\",\"token\":\"hello\"}"))
        XCTAssertEqual(try parser.consume(line: ""), .token("hello"))
    }

    func testParsesBackendDeltaEvent() throws {
        let event = try parseFrame([
            "event: delta",
            "data: {\"text\":\"你好\"}",
            ""
        ])

        XCTAssertEqual(event, .token("你好"))
    }

    func testParsesBackendTokenPayloadWithChineseText() throws {
        let event = try parseFrame([
            "event: token",
            "data: {\"type\": \"token\", \"token\": \"为\", \"delta\": \"为\", \"text\": \"为\"}",
            ""
        ])

        XCTAssertEqual(event, .token("为"))
    }

    func testParsesCRLFDelimitedBackendTokenPayload() throws {
        let event = try parseFrame([
            "event: token\r",
            "data: {\"type\": \"token\", \"token\": \"按\", \"delta\": \"按\", \"text\": \"按\"}\r",
            "\r"
        ])

        XCTAssertEqual(event, .token("按"))
    }

    func testDoesNotConcatenateCRLFDelimitedBackendFrames() throws {
        XCTAssertNil(try parser.consume(line: "event: token\r"))
        XCTAssertNil(try parser.consume(line: "data: {\"type\":\"token\",\"token\":\"按\",\"delta\":\"按\",\"text\":\"按\"}\r"))
        XCTAssertEqual(try parser.consume(line: "\r"), .token("按"))

        XCTAssertNil(try parser.consume(line: "event: token\r"))
        XCTAssertNil(try parser.consume(line: "data: {\"type\":\"token\",\"token\":\"需\",\"delta\":\"需\",\"text\":\"需\"}\r"))
        XCTAssertEqual(try parser.consume(line: "\r"), .token("需"))
    }

    func testParsesBareJSONTokenLine() throws {
        let event = try parseFrame([
            "{\"type\":\"token\",\"token\":\"好\",\"delta\":\"好\",\"text\":\"好\"}",
            ""
        ])

        XCTAssertEqual(event, .token("好"))
    }

    func testIgnoresBackendMetaEvent() throws {
        let event = try parseFrame([
            "event: meta",
            "data: {\"session_id\":\"session-1\",\"retrieval_source\":\"hybrid\",\"warnings\":[]}",
            ""
        ])

        XCTAssertNil(event)
    }

    func testThrowsMalformedEventForInvalidJSON() {
        XCTAssertThrowsError(try parseFrame(["data: {not-json}", ""])) { error in
            XCTAssertEqual(error as? ChatServiceError, .malformedEvent("{not-json}"))
        }
    }

    func testIgnoresCommentsAndUnknownFields() throws {
        let event = try parseFrame([
            ": keepalive",
            "id: abc",
            "data: {\"type\":\"done\",\"extra\":true}",
            ""
        ])

        XCTAssertEqual(event, .done(messageID: nil))
    }

    func testParsesDocumentedCartEventWithProductIDsOnly() throws {
        let event = try parseFrame([
            "event: cart",
            "data: {\"items\":[{\"product_id\":\"BOOT-1\",\"quantity\":1}],\"summary\":\"Added to cart\"}",
            ""
        ])

        XCTAssertEqual(event, .cartStatus(summary: "Added to cart"))
    }

    func testParsesComparisonEventWithProductMetadata() throws {
        let event = try parseFrame([
            "event: comparison",
            "data: {\"type\":\"comparison\",\"products\":[{\"product_id\":\"SUN-1\",\"title\":\"珂润润浸保湿防晒乳\",\"brand\":\"珂润\",\"category\":\"防晒\",\"sub_category\":\"防晒乳\",\"base_price\":158,\"image_path\":\"images/sun-1.jpg\",\"spec\":\"SPF50+ PA+++ 60ml\",\"rating\":4.8,\"sales\":\"10万+\",\"pros\":[\"无酒精无香精\"],\"cons\":[\"轻微泛白\"]}]}",
            ""
        ])

        XCTAssertEqual(event, .comparison([
            Product(
                id: "SUN-1",
                title: "珂润润浸保湿防晒乳",
                brand: "珂润",
                category: "防晒",
                subCategory: "防晒乳",
                basePrice: Decimal(158),
                imagePath: "images/sun-1.jpg",
                spec: "SPF50+ PA+++ 60ml",
                rating: 4.8,
                sales: "10万+",
                pros: ["无酒精无香精"],
                cons: ["轻微泛白"]
            )
        ]))
    }

    private func parseFrame(_ lines: [String]) throws -> ChatStreamEvent? {
        var result: ChatStreamEvent?
        for line in lines {
            if let event = try parser.consume(line: line) {
                result = event
            }
        }
        return result
    }
}

@available(iOS 17.0, macOS 13.0, *)
final class SSEChatServiceIntegrationTests: XCTestCase {
    override func tearDown() {
        StubURLProtocol.requestHandler = nil
        super.tearDown()
    }

    func testServiceStreamsBackendSSEIntoFrontendEvents() async throws {
        let body = [
            "event: token",
            "data: {\"type\":\"token\",\"token\":\"按\",\"delta\":\"按\",\"text\":\"按\"}",
            "",
            "event: products",
            "data: {\"type\":\"products\",\"products\":[{\"product_id\":\"p_beauty_007\",\"title\":\"薇诺娜舒敏保湿特护霜\",\"brand\":\"薇诺娜\",\"category\":\"美妆护肤\",\"sub_category\":\"面霜\",\"base_price\":89.0,\"price_label\":\"89元起（15g 体验装）\",\"price_summary\":\"15g 体验装 89元；50g 标准装 268元\",\"image_path\":\"1_美妆护肤/images/p_beauty_007_live.jpg\",\"reason\":\"适合敏感肌\"}],\"items\":[{\"product_id\":\"p_beauty_007\",\"title\":\"薇诺娜舒敏保湿特护霜\",\"brand\":\"薇诺娜\",\"category\":\"美妆护肤\",\"sub_category\":\"面霜\",\"base_price\":89.0,\"price_label\":\"89元起（15g 体验装）\",\"price_summary\":\"15g 体验装 89元；50g 标准装 268元\",\"image_path\":\"1_美妆护肤/images/p_beauty_007_live.jpg\",\"reason\":\"适合敏感肌\"}]}",
            "",
            "event: done",
            "data: {\"type\":\"done\",\"message_id\":\"msg-1\"}",
            ""
        ].joined(separator: "\r\n")

        StubURLProtocol.requestHandler = { request in
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "text/event-stream")

            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "text/event-stream; charset=utf-8"]
            )!
            return (response, Data(body.utf8))
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let service = SSEChatService(
            endpointURL: URL(string: "http://127.0.0.1:8000/api/chat/stream")!,
            session: session
        )

        let request = ChatRequest(
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000040")!,
            message: "推荐一个适合敏感肌的保湿护肤品"
        )

        var events: [ChatStreamEvent] = []
        for try await event in service.streamChat(for: request) {
            events.append(event)
        }

        XCTAssertEqual(events, [
            .token("按"),
            .products([
                Product(
                    id: "p_beauty_007",
                    title: "薇诺娜舒敏保湿特护霜",
                    brand: "薇诺娜",
                    category: "美妆护肤",
                    subCategory: "面霜",
                    basePrice: Decimal(string: "89.0")!,
                    priceLabel: "89元起（15g 体验装）",
                    priceSummary: "15g 体验装 89元；50g 标准装 268元",
                    imagePath: "1_美妆护肤/images/p_beauty_007_live.jpg",
                    reason: "适合敏感肌"
                )
            ]),
            .done(messageID: "msg-1")
        ])
    }

    func testServiceStreamsFromConfiguredBackendWhenEnabled() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let urlString = environment["ECOMMERCE_GUIDE_INTEGRATION_URL"],
              let url = URL(string: urlString) else {
            throw XCTSkip("Set ECOMMERCE_GUIDE_INTEGRATION_URL to run the real backend streaming test.")
        }

        let service = SSEChatService(endpointURL: url)
        let request = ChatRequest(
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000041")!,
            message: "推荐一个适合敏感肌的保湿护肤品，cheaper is better"
        )

        var answer = ""
        var products: [Product] = []
        var sawDone = false

        for try await event in service.streamChat(for: request) {
            switch event {
            case .token(let token):
                answer += token
            case .products(let streamedProducts):
                products = streamedProducts
            case .comparison:
                continue
            case .done:
                sawDone = true
            case .cartUpdated, .cartStatus:
                continue
            }
        }

        XCTAssertFalse(answer.isEmpty)
        XCTAssertFalse(products.isEmpty)
        XCTAssertTrue(answer.contains("89元起（15g 体验装）"))
        XCTAssertTrue(answer.contains("50g 标准装 268元"))
        XCTAssertEqual(products.first?.formattedPrice, "89元起（15g 体验装）")
        XCTAssertEqual(products.first?.priceSummary, "15g 体验装 89元；50g 标准装 268元")
        XCTAssertTrue(sawDone)
    }
}

private final class StubURLProtocol: URLProtocol {
    static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let requestHandler = Self.requestHandler else {
            client?.urlProtocol(self, didFailWithError: ChatServiceError.invalidResponse)
            return
        }

        do {
            let (response, data) = try requestHandler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
