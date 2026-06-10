import Foundation

@MainActor
@available(iOS 17.0, macOS 13.0, *)
public final class ChatViewModel: ObservableObject {
    @Published public var timeline: [ChatTimelineItem]
    @Published public var draftMessage: String
    @Published public var cartItems: [CartItem]
    @Published public var isSending: Bool
    @Published public var isListening: Bool
    @Published public var isAssistantSpeechEnabled: Bool
    @Published public var assistantSpeechPlaybackState: TextToSpeechPlaybackState
    @Published public var activeSpeechMessageID: UUID?
    @Published public var errorMessage: String?
    // True while a plan step that produces product cards is running and no cards have arrived yet,
    // so the chat can show a card skeleton that never lies about what is coming.
    @Published public private(set) var isAwaitingCards = false
    // The shipping address shown in the order card. Sent with every request so the order carries it,
    // and re-synced from the server's order whenever one arrives (e.g. after a conversational change).
    @Published public var shippingAddress: String = "北京市朝阳区望京SOHO T1 12层"

    public let conversationID: UUID

    private let service: any ChatService
    private let speechRecognitionService: any SpeechRecognitionService
    private let textToSpeechService: any TextToSpeechService
    private var streamTask: Task<Void, Never>?
    private var speechTask: Task<Void, Never>?
    private var lastSubmittedMessage: String?
    private var streamingMessageID: UUID?
    private var currentPlanTimelineID: UUID?
    private var hasReceivedCardsThisTurn = false
    // Plan step actions that produce product cards; aligned with server/planner.py PlanAction.
    private static let cardActions: Set<String> = ["product_search", "select_products", "comparison"]
    private var pendingSpeechMessage: ChatMessage?

    public init(
        service: any ChatService = MockChatService(),
        speechRecognitionService: any SpeechRecognitionService = DefaultSpeechRecognitionService(),
        textToSpeechService: (any TextToSpeechService)? = nil,
        isAssistantSpeechEnabled: Bool = true,
        conversationID: UUID = UUID(),
        timeline: [ChatTimelineItem] = ChatViewModel.initialTimeline
    ) {
        self.service = service
        self.speechRecognitionService = speechRecognitionService
        self.textToSpeechService = textToSpeechService ?? DefaultTextToSpeechService()
        self.conversationID = conversationID
        self.timeline = timeline
        self.draftMessage = ""
        self.cartItems = []
        self.isSending = false
        self.isListening = false
        self.isAssistantSpeechEnabled = isAssistantSpeechEnabled
        self.assistantSpeechPlaybackState = .idle
        self.activeSpeechMessageID = nil
        self.textToSpeechService.setPlaybackStateHandler { [weak self] state in
            self?.assistantSpeechPlaybackState = state
            if state == .idle {
                self?.activeSpeechMessageID = nil
            }
        }
        self.textToSpeechService.prepare(Self.welcomeMessageText)
    }

    deinit {
        streamTask?.cancel()
        speechTask?.cancel()
        speechRecognitionService.stopRecognition()
        let textToSpeechService = textToSpeechService
        Task { @MainActor in
            textToSpeechService.stopSpeaking()
        }
    }

    public func sendDraftMessage() {
        send(message: draftMessage)
    }

    public func sendPhoto(imageData: Data, caption: String) {
        let trimmed = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        let message = trimmed.isEmpty ? "找同款" : trimmed
        send(message: message, imageData: imageData)
    }

    public func retryLastMessage() {
        guard let lastSubmittedMessage else {
            return
        }

        send(message: lastSubmittedMessage)
    }

    /// Send a fixed reply on the user's behalf, e.g. the order card's 确认 / 取消订单 buttons.
    public func sendQuickReply(_ text: String) {
        send(message: text)
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
        isAwaitingCards = false
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

    public func speakAssistantMessage(_ message: ChatMessage) {
        guard message.role == .assistant, !message.isStreaming else {
            return
        }

        speak(message, automatically: false)
    }

    public func stopAssistantSpeech() {
        textToSpeechService.stopSpeaking()
    }

    public func toggleAssistantSpeechEnabled() {
        isAssistantSpeechEnabled.toggle()
        if !isAssistantSpeechEnabled {
            textToSpeechService.stopSpeaking()
        }
    }

    public func speechPlaybackState(for message: ChatMessage) -> TextToSpeechPlaybackState? {
        guard activeSpeechMessageID == message.id,
              assistantSpeechPlaybackState != .idle else {
            return nil
        }

        return assistantSpeechPlaybackState
    }

    private func send(message: String, imageData: Data? = nil) {
        if isListening {
            stopVoiceInput()
        }
        textToSpeechService.stopSpeaking()
        activeSpeechMessageID = nil
        pendingSpeechMessage = nil

        let trimmedMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedMessage.isEmpty, !isSending else {
            return
        }

        draftMessage = ""
        errorMessage = nil
        lastSubmittedMessage = trimmedMessage
        isSending = true
        currentPlanTimelineID = nil
        hasReceivedCardsThisTurn = false
        isAwaitingCards = false

        let assistantID = UUID()
        streamingMessageID = assistantID

        timeline.removeAll { item in
            if case .error = item {
                return true
            }
            return false
        }

        timeline.append(.message(ChatMessage(role: .user, text: trimmedMessage, imageData: imageData)))
        timeline.append(.message(ChatMessage(id: assistantID, role: .assistant, text: "", isStreaming: true)))

        let request = ChatRequest(
            conversationID: conversationID,
            message: trimmedMessage,
            cartItems: cartItems,
            recentProductIDs: recentProductIDs,
            imageData: imageData,
            address: shippingAddress
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
            isAwaitingCards = !hasReceivedCardsThisTurn && steps.contains {
                Self.cardActions.contains($0.action) && $0.status == "running"
            }
        case .products(let products):
            finishStreamingMessage()
            hasReceivedCardsThisTurn = true
            isAwaitingCards = false
            timeline.append(.products(id: UUID(), products: products))
        case .comparison(let products):
            finishStreamingMessage()
            hasReceivedCardsThisTurn = true
            isAwaitingCards = false
            timeline.append(.comparison(id: UUID(), comparison: products))
        case .cartUpdated(let items, let summary):
            cartItems = items
            timeline.append(.cartStatus(id: UUID(), text: summary))
        case .cartStatus(let summary):
            timeline.append(.cartStatus(id: UUID(), text: summary))
        case .orderStatus(let order):
            // Keep the editable field in step with the order (e.g. after a conversational "改地址").
            if !order.address.isEmpty {
                shippingAddress = order.address
            }
            timeline.append(.orderStatus(id: UUID(), order: order))
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

        if message.text.isEmpty && message.imageData == nil {
            // The placeholder never received any text (e.g. a cards-only reply); drop it rather
            // than leaving an empty bubble in the timeline.
            timeline.remove(at: index)
        } else {
            message.isStreaming = false
            timeline[index] = .message(message)
            if message.role == .assistant {
                pendingSpeechMessage = message
            }
        }
        self.streamingMessageID = nil
    }

    private func handle(_ error: Error) {
        isSending = false
        isAwaitingCards = false
        finishStreamingMessage()
        pendingSpeechMessage = nil

        let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        errorMessage = message
        timeline.append(.error(id: UUID(), message: message))
        streamTask = nil
    }

    private func finishAfterCancellation() {
        isSending = false
        isAwaitingCards = false
        finishStreamingMessage()
        pendingSpeechMessage = nil
        streamTask = nil
    }

    private func finishCompletedStream() {
        isSending = false
        isAwaitingCards = false
        finishStreamingMessage()
        speakPendingAssistantMessage()
        streamTask = nil
    }

    private func speakPendingAssistantMessage() {
        guard let message = pendingSpeechMessage else {
            return
        }

        pendingSpeechMessage = nil
        speak(message, automatically: true)
    }

    private func speak(_ message: ChatMessage, automatically: Bool) {
        guard !automatically || isAssistantSpeechEnabled else {
            return
        }

        activeSpeechMessageID = message.id
        textToSpeechService.speak(message.text)
        if assistantSpeechPlaybackState != .idle {
            activeSpeechMessageID = message.id
        }
    }

    nonisolated public static var initialTimeline: [ChatTimelineItem] {
        [
            .message(ChatMessage(
                role: .assistant,
                text: welcomeMessageText
            ))
        ]
    }

    nonisolated private static let welcomeMessageText = "你好，我是你的 AI 购物助手。告诉我你想买什么，我会帮你对比商品、解释取舍，并整理好购物车。"

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
