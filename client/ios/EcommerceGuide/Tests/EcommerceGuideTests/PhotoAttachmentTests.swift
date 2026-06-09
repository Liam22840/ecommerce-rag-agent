import XCTest
@testable import EcommerceGuide

final class PhotoAttachmentTests: XCTestCase {
    func testRequestEncodesImageAttachmentAsBase64() throws {
        let imageData = Data([0xFF, 0xD8, 0xFF, 0xD9])
        let request = ChatRequest(
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000099")!,
            message: "找同款",
            imageData: imageData
        )

        let encoded = try JSONEncoder().encode(ChatRequestPayload(request: request))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let attachments = try XCTUnwrap(json["attachments"] as? [[String: Any]])

        XCTAssertEqual(attachments.count, 1)
        XCTAssertEqual(attachments[0]["type"] as? String, "image")
        XCTAssertEqual(attachments[0]["mime"] as? String, "image/jpeg")
        XCTAssertEqual(attachments[0]["data"] as? String, imageData.base64EncodedString())
    }

    func testRequestWithoutImageEncodesEmptyAttachments() throws {
        let request = ChatRequest(conversationID: UUID(), message: "推荐面霜")
        let encoded = try JSONEncoder().encode(ChatRequestPayload(request: request))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual((json["attachments"] as? [[String: Any]])?.count, 0)
    }
}
