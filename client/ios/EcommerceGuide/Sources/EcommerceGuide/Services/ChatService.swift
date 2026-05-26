import Foundation

public struct ChatRequest: Equatable, Sendable {
    public let conversationID: UUID
    public let message: String
    public let cartItems: [CartItem]

    public init(conversationID: UUID, message: String, cartItems: [CartItem] = []) {
        self.conversationID = conversationID
        self.message = message
        self.cartItems = cartItems
    }
}

@available(iOS 17.0, macOS 13.0, *)
public protocol ChatService: Sendable {
    func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error>
}
