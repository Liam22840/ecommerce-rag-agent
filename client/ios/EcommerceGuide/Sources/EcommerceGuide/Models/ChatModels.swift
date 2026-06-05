import Foundation

public enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
}

public struct ChatMessage: Identifiable, Equatable, Sendable {
    public let id: UUID
    public let role: ChatRole
    public var text: String
    public var isStreaming: Bool

    public init(
        id: UUID = UUID(),
        role: ChatRole,
        text: String,
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
    }
}

public enum ChatTimelineItem: Identifiable, Equatable, Sendable {
    case message(ChatMessage)
    case products(id: UUID, products: [Product])
    case comparison(id: UUID, products: [Product])
    case cartStatus(id: UUID, text: String)
    case error(id: UUID, message: String)

    public var id: UUID {
        switch self {
        case .message(let message):
            message.id
        case .products(let id, _),
             .comparison(let id, _),
             .cartStatus(let id, _),
             .error(let id, _):
            id
        }
    }
}

public enum ChatStreamEvent: Equatable, Sendable {
    case token(String)
    case products([Product])
    case comparison([Product])
    case cartUpdated([CartItem], summary: String)
    case cartStatus(summary: String)
    case done(messageID: String?)
}
