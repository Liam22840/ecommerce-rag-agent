import Foundation

@MainActor
@available(iOS 17.0, macOS 13.0, *)
public final class ChatViewModel: ObservableObject {
    @Published public var timeline: [ChatTimelineItem]
    @Published public var draftMessage: String
    @Published public var cartItems: [CartItem]
    @Published public var isSending: Bool
    @Published public var isListening: Bool
    @Published public var errorMessage: String?

    public let conversationID: UUID

    private let service: any ChatService
    private let speechRecognitionService: any SpeechRecognitionService
    private var streamTask: Task<Void, Never>?
    private var speechTask: Task<Void, Never>?
    private var lastSubmittedMessage: String?
    private var streamingMessageID: UUID?
    private var currentPlanTimelineID: UUID?

    public init(
        service: any ChatService = MockChatService(),
        speechRecognitionService: any SpeechRecognitionService = DefaultSpeechRecognitionService(),
        conversationID: UUID = UUID(),
        timeline: [ChatTimelineItem] = ChatViewModel.initialTimeline
    ) {
        self.service = service
        self.speechRecognitionService = speechRecognitionService
        self.conversationID = conversationID
        self.timeline = timeline
        self.draftMessage = ""
        self.cartItems = []
        self.isSending = false
        self.isListening = false
    }

    deinit {
        streamTask?.cancel()
        speechTask?.cancel()
        speechRecognitionService.stopRecognition()
    }

    public func sendDraftMessage() {
        send(message: draftMessage)
    }

    public func retryLastMessage() {
        guard let lastSubmittedMessage else {
            return
        }

        send(message: lastSubmittedMessage)
    }

    public func addToCart(product: Product) {
        if let index = cartItems.firstIndex(where: { $0.product.id == product.id }) {
            cartItems[index].quantity += 1
        } else {
            cartItems.append(CartItem(product: product))
        }

        timeline.append(.cartStatus(
            id: UUID(),
            text: "已将「\(product.title)」加入购物车。"
        ))
    }

    public func updateCartItem(productID: String, delta: Int) {
        guard let index = cartItems.firstIndex(where: { $0.product.id == productID }) else {
            return
        }

        let nextQuantity = cartItems[index].quantity + delta
        if nextQuantity > 0 {
            cartItems[index].quantity = nextQuantity
        } else {
            cartItems.remove(at: index)
        }
    }

    public func removeFromCart(productID: String) {
        cartItems.removeAll { $0.product.id == productID }
    }

    public func cancelStreaming() {
        streamTask?.cancel()
        streamTask = nil
        isSending = false
        finishStreamingMessage()
    }

    public func toggleVoiceInput() {
        guard !isSending else {
            return
        }

        if isListening {
            stopVoiceInput(shouldSendTranscript: true)
        } else {
            startVoiceInput()
        }
    }

    public func stopVoiceInput(shouldSendTranscript: Bool = false) {
        guard isListening else {
            return
        }

        isListening = false
        speechTask?.cancel()
        speechTask = nil
        speechRecognitionService.stopRecognition()

        if shouldSendTranscript {
            sendDraftMessage()
        }
    }

    private func send(message: String) {
        if isListening {
            stopVoiceInput()
        }

        let trimmedMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedMessage.isEmpty, !isSending else {
            return
        }

        draftMessage = ""
        errorMessage = nil
        lastSubmittedMessage = trimmedMessage
        isSending = true
        currentPlanTimelineID = nil

        let assistantID = UUID()
        streamingMessageID = assistantID

        timeline.removeAll { item in
            if case .error = item {
                return true
            }
            return false
        }

        timeline.append(.message(ChatMessage(role: .user, text: trimmedMessage)))
        timeline.append(.message(ChatMessage(id: assistantID, role: .assistant, text: "", isStreaming: true)))

        let request = ChatRequest(
            conversationID: conversationID,
            message: trimmedMessage,
            cartItems: cartItems,
            recentProductIDs: recentProductIDs
        )

        streamTask?.cancel()
        streamTask = Task { [weak self, service] in
            do {
                for try await event in service.streamChat(for: request) {
                    try Task.checkCancellation()
                    self?.reduce(event)
                }
                self?.finishCompletedStream()
            } catch is CancellationError {
                self?.finishAfterCancellation()
            } catch {
                self?.handle(error)
            }
        }
    }

    private func startVoiceInput() {
        guard !isListening, !isSending else {
            return
        }

        errorMessage = nil
        draftMessage = ""
        isListening = true
        speechTask?.cancel()

        speechTask = Task { @MainActor [weak self, speechRecognitionService] in
            do {
                for try await update in speechRecognitionService.startMandarinRecognition() {
                    try Task.checkCancellation()
                    self?.applySpeechRecognition(update)
                }
                self?.finishVoiceInput()
            } catch is CancellationError {
                self?.finishVoiceInput()
            } catch {
                self?.handleSpeechRecognition(error)
            }
        }
    }

    private func applySpeechRecognition(_ update: SpeechRecognitionUpdate) {
        draftMessage = update.transcript

        if update.isFinal {
            stopVoiceInput(shouldSendTranscript: true)
        }
    }

    private func finishVoiceInput() {
        guard isListening else {
            return
        }

        isListening = false
        speechTask = nil
    }

    private func handleSpeechRecognition(_ error: Error) {
        isListening = false
        speechTask = nil
        speechRecognitionService.stopRecognition()

        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        errorMessage = message
        timeline.append(.error(id: UUID(), message: message))
    }

    private func reduce(_ event: ChatStreamEvent) {
        switch event {
        case .token(let token):
            appendToken(token)
        case .plan(let steps):
            finishStreamingMessage()
            upsertPlan(steps)
        case .products(let products):
            finishStreamingMessage()
            timeline.append(.products(id: UUID(), products: products))
        case .comparison(let products):
            finishStreamingMessage()
            timeline.append(.comparison(id: UUID(), comparison: products))
        case .cartUpdated(let items, let summary):
            cartItems = items
            timeline.append(.cartStatus(id: UUID(), text: summary))
        case .cartStatus(let summary):
            timeline.append(.cartStatus(id: UUID(), text: summary))
        case .orderStatus(let summary):
            timeline.append(.orderStatus(id: UUID(), text: summary))
        case .done:
            finishCompletedStream()
        }
    }

    private func upsertPlan(_ steps: [PlanStep]) {
        if let currentPlanTimelineID,
           let index = timeline.firstIndex(where: { $0.id == currentPlanTimelineID }),
           case .plan(let id, _) = timeline[index] {
            timeline[index] = .plan(id: id, steps: steps)
            return
        }

        let id = UUID()
        currentPlanTimelineID = id
        timeline.append(.plan(id: id, steps: steps))
    }

    private func appendToken(_ token: String) {
        guard let streamingMessageID,
              let index = timeline.firstIndex(where: { $0.id == streamingMessageID }),
              case .message(var message) = timeline[index] else {
            let message = ChatMessage(role: .assistant, text: token, isStreaming: true)
            streamingMessageID = message.id
            timeline.append(.message(message))
            return
        }

        message.text += token
        timeline[index] = .message(message)
    }

    private func finishStreamingMessage() {
        guard let streamingMessageID,
              let index = timeline.firstIndex(where: { $0.id == streamingMessageID }),
              case .message(var message) = timeline[index] else {
            self.streamingMessageID = nil
            return
        }

        message.isStreaming = false
        timeline[index] = .message(message)
        self.streamingMessageID = nil
    }

    private func handle(_ error: Error) {
        isSending = false
        finishStreamingMessage()

        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        errorMessage = message
        timeline.append(.error(id: UUID(), message: message))
        streamTask = nil
    }

    private func finishAfterCancellation() {
        isSending = false
        finishStreamingMessage()
        streamTask = nil
    }

    private func finishCompletedStream() {
        isSending = false
        finishStreamingMessage()
        streamTask = nil
    }

    nonisolated public static var initialTimeline: [ChatTimelineItem] {
        [
            .message(ChatMessage(
                role: .assistant,
                text: "你好，我是你的 AI 购物助手。告诉我你想买什么，我会帮你对比商品、解释取舍，并整理好购物车。"
            ))
        ]
    }

    private var recentProductIDs: [String] {
        var ids: [String] = []
        for item in timeline.reversed() {
            switch item {
            case .products(_, let products):
                ids.append(contentsOf: products.map(\.id))
            case .comparison(_, let comparison):
                ids.append(contentsOf: comparison.products.map(\.id))
            case .message, .plan, .cartStatus, .orderStatus, .error:
                continue
            }
            if ids.count >= 10 {
                break
            }
        }
        var seen = Set<String>()
        return ids.filter { id in
            guard !seen.contains(id) else {
                return false
            }
            seen.insert(id)
            return true
        }
    }
}
