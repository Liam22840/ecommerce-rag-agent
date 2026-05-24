import Foundation

@available(iOS 17.0, macOS 14.0, *)
public struct SSEChatService: ChatService {
    public let endpointURL: URL

    private let session: URLSession

    public init(
        endpointURL: URL = SSEChatService.defaultEndpointURL,
        session: URLSession = .shared
    ) {
        self.endpointURL = endpointURL
        self.session = session
    }

    public func streamChat(for request: ChatRequest) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let endpointURL = endpointURL
            let session = session

            let task = Task {
                do {
                    var urlRequest = URLRequest(url: endpointURL)
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    urlRequest.httpBody = try JSONEncoder().encode(ChatRequestPayload(request: request))

                    let (bytes, response) = try await session.bytes(for: urlRequest)

                    guard let httpResponse = response as? HTTPURLResponse,
                          (200..<300).contains(httpResponse.statusCode) else {
                        throw ChatServiceError.invalidResponse
                    }

                    var parser = SSEEventParser()

                    for try await line in bytes.lines {
                        if let event = try parser.consume(line: line) {
                            continuation.yield(event)
                        }
                    }

                    if let event = try parser.finish() {
                        continuation.yield(event)
                    }

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

    public static var defaultEndpointURL: URL {
        if let value = UserDefaults.standard.string(forKey: "EcommerceGuideBackendURL"),
           let url = URL(string: value) {
            return url
        }

        return URL(string: "http://127.0.0.1:8000/api/v1/chat/stream")!
    }
}

struct SSEEventParser {
    private var eventName: String?
    private var dataLines: [String] = []

    mutating func consume(line: String) throws -> ChatStreamEvent? {
        guard !line.isEmpty else {
            return try flush()
        }

        if line.hasPrefix(":") {
            return nil
        } else if line.hasPrefix("event:") {
            eventName = line.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
        } else if line.hasPrefix("data:") {
            let value = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
            dataLines.append(value)
        }

        return nil
    }

    mutating func finish() throws -> ChatStreamEvent? {
        try flush()
    }

    private mutating func flush() throws -> ChatStreamEvent? {
        guard !dataLines.isEmpty else {
            return nil
        }

        let payload = dataLines.joined(separator: "\n")
        let eventName = eventName
        self.eventName = nil
        dataLines.removeAll(keepingCapacity: true)

        if payload == "[DONE]" {
            return .done(messageID: nil)
        }

        guard let data = payload.data(using: .utf8) else {
            throw ChatServiceError.malformedEvent(payload)
        }

        do {
            let decoded = try JSONDecoder().decode(StreamEventPayload.self, from: data)
            return try decoded.streamEvent(fallbackType: eventName)
        } catch {
            throw ChatServiceError.malformedEvent(payload)
        }
    }
}

private struct StreamEventPayload: Decodable {
    let type: String?
    let event: String?
    let token: String?
    let text: String?
    let delta: String?
    let products: [Product]?
    let items: [Product]?
    let cartItems: [CartItemPayload]?
    let summary: String?
    let messageID: String?

    enum CodingKeys: String, CodingKey {
        case type
        case event
        case token
        case text
        case delta
        case products
        case items
        case cartItems
        case cartItemsSnake = "cart_items"
        case summary
        case messageID
        case messageIDSnake = "message_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(String.self, forKey: .type)
        self.event = try container.decodeIfPresent(String.self, forKey: .event)
        self.token = try container.decodeIfPresent(String.self, forKey: .token)
        self.text = try container.decodeIfPresent(String.self, forKey: .text)
        self.delta = try container.decodeIfPresent(String.self, forKey: .delta)
        self.products = try container.decodeIfPresent([Product].self, forKey: .products)
        self.items = try? container.decodeIfPresent([Product].self, forKey: .items)
        self.cartItems = try container.decodeIfPresent([CartItemPayload].self, forKey: .cartItems)
            ?? container.decodeIfPresent([CartItemPayload].self, forKey: .cartItemsSnake)
            ?? (try? container.decodeIfPresent([CartItemPayload].self, forKey: .items))
        self.summary = try container.decodeIfPresent(String.self, forKey: .summary)
        self.messageID = try container.decodeIfPresent(String.self, forKey: .messageID)
            ?? container.decodeIfPresent(String.self, forKey: .messageIDSnake)
    }

    func streamEvent(fallbackType: String?) throws -> ChatStreamEvent {
        switch type ?? event ?? fallbackType {
        case "token":
            return .token(token ?? delta ?? text ?? "")
        case "products":
            return .products(products ?? items ?? [])
        case "cart", "cart_updated", "cartUpdated":
            let summary = summary ?? "Cart updated."
            guard let cartItems else {
                return .cartStatus(summary: summary)
            }

            let parsedItems = cartItems.compactMap(\.cartItem)
            guard parsedItems.count == cartItems.count else {
                return .cartStatus(summary: summary)
            }

            return .cartUpdated(parsedItems, summary: summary)
        case "done":
            return .done(messageID: messageID)
        default:
            throw ChatServiceError.malformedEvent(type ?? event ?? "unknown")
        }
    }
}

private struct CartItemPayload: Codable {
    let product: Product?
    let productID: String?
    let quantity: Int

    enum CodingKeys: String, CodingKey {
        case product
        case productID = "product_id"
        case quantity
    }

    var cartItem: CartItem? {
        guard let product else {
            return nil
        }

        return CartItem(product: product, quantity: quantity)
    }

    init(cartItem: CartItem) {
        self.product = cartItem.product
        self.productID = cartItem.product.id
        self.quantity = cartItem.quantity
    }
}

private struct ChatRequestPayload: Encodable {
    let conversationID: UUID
    let message: String
    let attachments: [String]
    let clientContext: ClientContextPayload

    init(request: ChatRequest) {
        self.conversationID = request.conversationID
        self.message = request.message
        self.attachments = []
        self.clientContext = ClientContextPayload(
            cartItems: request.cartItems.map(CartItemPayload.init(cartItem:))
        )
    }

    enum CodingKeys: String, CodingKey {
        case conversationID = "conversation_id"
        case message
        case attachments
        case clientContext = "client_context"
    }
}

private struct ClientContextPayload: Encodable {
    let cartItems: [CartItemPayload]

    enum CodingKeys: String, CodingKey {
        case cartItems = "cart_items"
    }
}
