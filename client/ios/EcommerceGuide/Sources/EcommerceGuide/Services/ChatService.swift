import Foundation

public struct ChatRequest: Equatable, Sendable {
    public let conversationID: UUID
    public let message: String
    public let cartItems: [CartItem]
    public let recentProductIDs: [String]
    public let compareProductIDs: [String]
    public let imageData: Data?

    public init(
        conversationID: UUID,
        message: String,
        cartItems: [CartItem] = [],
        recentProductIDs: [String] = [],
        compareProductIDs: [String] = [],
        imageData: Data? = nil
    ) {
        self.conversationID = conversationID
        self.message = message
        self.cartItems = cartItems
        self.recentProductIDs = recentProductIDs
        self.compareProductIDs = compareProductIDs
        self.imageData = imageData
    }
}

@available(iOS 17.0, macOS 13.0, *)
public protocol ChatService: Sendable {
    func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error>
}
