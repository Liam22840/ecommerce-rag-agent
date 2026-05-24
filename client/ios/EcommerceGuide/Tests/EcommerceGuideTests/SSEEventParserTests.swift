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
            "data: {\"items\":[{\"product_id\":\"BOOT-1\",\"title\":\"Summit Boot\",\"brand\":\"Northstar\",\"category\":\"Shoes\",\"sub_category\":\"Boots\",\"base_price\":189.00,\"image_path\":\"images/boot-1.jpg\",\"reason\":\"Good grip\"}]}",
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
