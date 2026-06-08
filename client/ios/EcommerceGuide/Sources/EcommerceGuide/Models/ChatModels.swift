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
    public var imageData: Data?

    public init(
        id: UUID = UUID(),
        role: ChatRole,
        text: String,
        isStreaming: Bool = false,
        imageData: Data? = nil
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
        self.imageData = imageData
    }
}

public enum ChatTimelineItem: Identifiable, Equatable, Sendable {
    case message(ChatMessage)
    case plan(id: UUID, steps: [PlanStep])
    case products(id: UUID, products: [Product])
    case comparison(id: UUID, comparison: ProductComparison)
    case cartStatus(id: UUID, text: String)
    case orderStatus(id: UUID, text: String)
    case error(id: UUID, message: String)

    public var id: UUID {
        switch self {
        case .message(let message):
            message.id
        case .plan(let id, _),
             .products(let id, _),
             .comparison(let id, _),
             .cartStatus(let id, _),
             .orderStatus(let id, _),
             .error(let id, _):
            id
        }
    }
}

public enum ChatStreamEvent: Equatable, Sendable {
    case token(String)
    case plan([PlanStep])
    case products([Product])
    case comparison(ProductComparison)
    case cartUpdated([CartItem], summary: String)
    case cartStatus(summary: String)
    case orderStatus(summary: String)
    case done(messageID: String?)
}

public struct PlanStep: Codable, Equatable, Sendable {
    public let stepID: String
    public let title: String
    public let action: String
    public let status: String
    public let summary: String?

    enum CodingKeys: String, CodingKey {
        case stepID = "step_id"
        case title
        case action
        case status
        case summary
    }

    public init(
        stepID: String,
        title: String,
        action: String,
        status: String = "pending",
        summary: String? = nil
    ) {
        self.stepID = stepID
        self.title = title
        self.action = action
        self.status = status
        self.summary = summary
    }
}

public struct ProductComparison: Codable, Equatable, Sendable {
    public let products: [Product]
    public let focus: [String]
    public let rows: [ComparisonRow]
    public let winnerProductID: String?
    public let recommendation: String?
    public let summary: String?
    public let clarification: String?

    enum CodingKeys: String, CodingKey {
        case products
        case focus
        case rows
        case winnerProductID = "winner_product_id"
        case recommendation
        case summary
        case clarification
    }

    public init(
        products: [Product],
        focus: [String] = [],
        rows: [ComparisonRow] = [],
        winnerProductID: String? = nil,
        recommendation: String? = nil,
        summary: String? = nil,
        clarification: String? = nil
    ) {
        self.products = products
        self.focus = focus
        self.rows = rows
        self.winnerProductID = winnerProductID
        self.recommendation = recommendation
        self.summary = summary
        self.clarification = clarification
    }
}

public struct ComparisonRow: Codable, Equatable, Sendable {
    public let dimension: String
    public let values: [ComparisonValue]
    public let winnerProductID: String?
    public let verdict: String

    enum CodingKeys: String, CodingKey {
        case dimension
        case values
        case winnerProductID = "winner_product_id"
        case verdict
    }

    public init(
        dimension: String,
        values: [ComparisonValue],
        winnerProductID: String? = nil,
        verdict: String
    ) {
        self.dimension = dimension
        self.values = values
        self.winnerProductID = winnerProductID
        self.verdict = verdict
    }
}

public struct ComparisonValue: Codable, Equatable, Sendable {
    public let productID: String
    public let value: String
    public let evidence: [String]
    public let confidence: String

    enum CodingKeys: String, CodingKey {
        case productID = "product_id"
        case value
        case evidence
        case confidence
    }

    public init(
        productID: String,
        value: String,
        evidence: [String] = [],
        confidence: String = "none"
    ) {
        self.productID = productID
        self.value = value
        self.evidence = evidence
        self.confidence = confidence
    }
}
