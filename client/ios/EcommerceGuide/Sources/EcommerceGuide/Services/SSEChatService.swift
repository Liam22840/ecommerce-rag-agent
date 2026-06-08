import Foundation

@available(iOS 17.0, macOS 13.0, *)
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
                    var lineScanner = SSELineScanner()

                    for try await byte in bytes {
                        guard let line = lineScanner.consume(byte: byte) else {
                            continue
                        }

                        if let event = try parser.consume(line: line) {
                            continuation.yield(event)
                        }
                    }

                    if let line = lineScanner.finish(),
                       let event = try parser.consume(line: line) {
                        continuation.yield(event)
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

        return URL(string: "http://127.0.0.1:8000/api/chat/stream")!
    }
}

struct SSELineScanner {
    private var buffer = Data()
    private var previousByteWasCarriageReturn = false

    mutating func consume(byte: UInt8) -> String? {
        if previousByteWasCarriageReturn {
            previousByteWasCarriageReturn = false
            if byte == Self.lineFeed {
                return nil
            }
        }

        switch byte {
        case Self.lineFeed:
            return flush()
        case Self.carriageReturn:
            previousByteWasCarriageReturn = true
            return flush()
        default:
            buffer.append(byte)
            return nil
        }
    }

    mutating func finish() -> String? {
        previousByteWasCarriageReturn = false
        guard !buffer.isEmpty else {
            return nil
        }

        return flush()
    }

    private mutating func flush() -> String {
        let line = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: true)
        return line
    }

    private static let lineFeed = UInt8(ascii: "\n")
    private static let carriageReturn = UInt8(ascii: "\r")
}

struct SSEEventParser {
    private var eventName: String?
    private var dataLines: [String] = []

    mutating func consume(line: String) throws -> ChatStreamEvent? {
        let line = line.trimmingCharacters(in: CharacterSet(charactersIn: "\r\n"))

        guard !line.isEmpty else {
            return try flush()
        }

        if line.hasPrefix(":") {
            return nil
        } else if line.hasPrefix("event:") {
            eventName = sseFieldValue(from: line, field: "event")
        } else if line.hasPrefix("data:") {
            dataLines.append(sseFieldValue(from: line, field: "data"))
        } else if line == "[DONE]" || line.hasPrefix("{") || line.hasPrefix("[") {
            dataLines.append(line)
        }

        return nil
    }

    mutating func finish() throws -> ChatStreamEvent? {
        try flush()
    }

    private mutating func flush() throws -> ChatStreamEvent? {
        defer {
            eventName = nil
            dataLines.removeAll(keepingCapacity: true)
        }

        guard !dataLines.isEmpty else {
            return nil
        }

        let payload = dataLines.joined(separator: "\n")
        let eventName = eventName

        if payload == "[DONE]" {
            return .done(messageID: nil)
        }

        guard let data = payload.data(using: .utf8) else {
            throw ChatServiceError.malformedEvent(payload)
        }

        do {
            if let event = try parsePrimitiveEvent(data: data, fallbackType: eventName) {
                return event
            }
        } catch {
            throw ChatServiceError.malformedEvent(payload)
        }

        do {
            let decoded = try JSONDecoder().decode(StreamEventPayload.self, from: data)
            return decoded.streamEvent(fallbackType: eventName)
        } catch {
            throw ChatServiceError.malformedEvent(payload)
        }
    }

    private func parsePrimitiveEvent(data: Data, fallbackType: String?) throws -> ChatStreamEvent? {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let payload = object as? [String: Any] else {
            return nil
        }

        let eventType = payload["type"] as? String
            ?? payload["event"] as? String
            ?? fallbackType

        switch eventType {
        case "token", "delta":
            let token = payload["token"] as? String
                ?? payload["delta"] as? String
                ?? payload["text"] as? String
                ?? ""
            return .token(token)
        case "comparison", "compare", "product_comparison", "productComparison":
            return nil
        case "done":
            return .done(messageID: payload["message_id"] as? String ?? payload["messageID"] as? String)
        case "meta", "metadata", "debug", "warning", "warnings":
            return nil
        default:
            return nil
        }
    }

    private func sseFieldValue(from line: String, field: String) -> String {
        var value = line.dropFirst(field.count + 1)
        if value.first == " " {
            value = value.dropFirst()
        }
        return String(value)
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
    let focus: [String]?
    let rows: [ComparisonRow]?
    let winnerProductID: String?
    let recommendation: String?
    let clarification: String?
    let steps: [PlanStep]?
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
        case focus
        case rows
        case winnerProductID = "winner_product_id"
        case recommendation
        case clarification
        case steps
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
        self.focus = try container.decodeIfPresent([String].self, forKey: .focus)
        self.rows = try container.decodeIfPresent([ComparisonRow].self, forKey: .rows)
        self.winnerProductID = try container.decodeIfPresent(String.self, forKey: .winnerProductID)
        self.recommendation = try container.decodeIfPresent(String.self, forKey: .recommendation)
        self.clarification = try container.decodeIfPresent(String.self, forKey: .clarification)
        self.steps = try container.decodeIfPresent([PlanStep].self, forKey: .steps)
        self.cartItems = try container.decodeIfPresent([CartItemPayload].self, forKey: .cartItems)
            ?? container.decodeIfPresent([CartItemPayload].self, forKey: .cartItemsSnake)
            ?? (try? container.decodeIfPresent([CartItemPayload].self, forKey: .items))
        self.summary = try container.decodeIfPresent(String.self, forKey: .summary)
        self.messageID = try container.decodeIfPresent(String.self, forKey: .messageID)
            ?? container.decodeIfPresent(String.self, forKey: .messageIDSnake)
    }

    func streamEvent(fallbackType: String?) -> ChatStreamEvent? {
        switch type ?? event ?? fallbackType {
        case "token", "delta":
            return .token(token ?? delta ?? text ?? "")
        case "plan":
            return .plan(steps ?? [])
        case "products":
            return .products(products ?? items ?? [])
        case "comparison", "compare", "product_comparison", "productComparison":
            return .comparison(ProductComparison(
                products: products ?? items ?? [],
                focus: focus ?? [],
                rows: rows ?? [],
                winnerProductID: winnerProductID,
                recommendation: recommendation,
                summary: summary,
                clarification: clarification
            ))
        case "cart", "cart_updated", "cartUpdated":
            let summary = summary ?? "购物车已更新。"
            guard let cartItems else {
                return .cartStatus(summary: summary)
            }

            let parsedItems = cartItems.compactMap(\.cartItem)
            guard parsedItems.count == cartItems.count else {
                return .cartStatus(summary: summary)
            }

            return .cartUpdated(parsedItems, summary: summary)
        case "order", "order_draft", "orderDraft", "order_submitted", "orderSubmitted":
            return .orderStatus(summary: summary ?? "订单状态已更新。")
        case "done":
            return .done(messageID: messageID)
        case "meta", "metadata", "debug", "warning", "warnings":
            return nil
        default:
            return nil
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
    let compareProductIDs: [String]
    let attachments: [String]
    let clientContext: ClientContextPayload

    init(request: ChatRequest) {
        self.conversationID = request.conversationID
        self.message = request.message
        self.compareProductIDs = request.compareProductIDs
        self.attachments = []
        self.clientContext = ClientContextPayload(
            cartItems: request.cartItems.map(CartItemPayload.init(cartItem:)),
            recentProductIDs: request.recentProductIDs,
            compareProductIDs: request.compareProductIDs
        )
    }

    enum CodingKeys: String, CodingKey {
        case conversationID = "conversation_id"
        case message
        case compareProductIDs = "compare_product_ids"
        case attachments
        case clientContext = "client_context"
    }
}

private struct ClientContextPayload: Encodable {
    let cartItems: [CartItemPayload]
    let recentProductIDs: [String]
    let compareProductIDs: [String]

    enum CodingKeys: String, CodingKey {
        case cartItems = "cart_items"
        case recentProductIDs = "recent_product_ids"
        case compareProductIDs = "compare_product_ids"
    }
}
