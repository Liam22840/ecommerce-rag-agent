import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
public struct ChatScreen: View {
    @StateObject private var viewModel: ChatViewModel
    @StateObject private var favourites = FavouritesStore()
    @State private var selectedProduct: Product?
    @State private var isCartPresented = false
    @State private var isFavouritesPresented = false
    @State private var isSettingsPresented = false
    @State private var flight: CartFlight?
    // Auto-scroll only follows the conversation while the user is at the bottom. Once they scroll up
    // to read, this goes false and the per-token scrollTo stops, so it never yanks them back.
    @State private var isPinnedToBottom = true
    @State private var viewportHeight: CGFloat = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let cameraAction: () -> Void
    private let photoLibraryAction: () -> Void
    private let checkoutAction: () -> Void

    private static let skeletonID = "products-skeleton"
    private static let thinkingID = "assistant-thinking"
    // A fixed invisible element pinned at the very end of the list. Scrolling to a stable id always
    // lands at the true bottom, unlike scrolling to the last timeline item — whose id changes every
    // turn and is often a just-inserted empty bubble, which makes LazyVStack jump to a wrong estimate.
    private static let bottomAnchorID = "chat-bottom-anchor"
    private static let scrollSpace = "chat-scroll-space"
    // The user counts as "following" the conversation while the bottom anchor is within this many
    // points of the viewport, so streaming keeps up but a small manual scroll up still detaches.
    private static let pinThreshold: CGFloat = 80

    @MainActor
    public init() {
        _viewModel = StateObject(wrappedValue: ChatViewModel())
        self.cameraAction = {}
        self.photoLibraryAction = {}
        self.checkoutAction = {}
    }

    @MainActor
    public init(
        viewModel: ChatViewModel,
        cameraAction: @escaping () -> Void = {},
        photoLibraryAction: @escaping () -> Void = {},
        checkoutAction: @escaping () -> Void = {}
    ) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.cameraAction = cameraAction
        self.photoLibraryAction = photoLibraryAction
        self.checkoutAction = checkoutAction
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ChatHeaderView(
                    cartItems: viewModel.cartItems,
                    favouritesCount: favourites.items.count,
                    settingsAction: { isSettingsPresented = true },
                    cartAction: { isCartPresented = true },
                    favouritesAction: { isFavouritesPresented = true }
                )

                Divider()
                    .overlay(GuideTheme.line)

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            Text("今天")
                                .font(.caption2)
                                .foregroundStyle(GuideTheme.tertiaryInk)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding(.bottom, 2)

                            ForEach(viewModel.timeline) { item in
                                ChatTimelineItemView(
                                    item: item,
                                    isOrderActionable: item.id == viewModel.latestOrderTimelineID && !viewModel.isSending,
                                    shippingAddress: $viewModel.shippingAddress,
                                    retryAction: { viewModel.retryLastMessage() },
                                    productAction: { selectedProduct = $0 },
                                    addToCartAction: { addToCart($0) },
                                    orderReplyAction: { viewModel.sendQuickReply($0) },
                                    speechPlaybackState: { viewModel.speechPlaybackState(for: $0) },
                                    speakAction: { viewModel.speakAssistantMessage($0) },
                                    stopSpeechAction: { viewModel.stopAssistantSpeech() }
                                )
                                .id(item.id)
                                .transition(GuideMotion.timelineInsertion(reduceMotion: reduceMotion))
                            }

                            if viewModel.isAwaitingCards {
                                ProductCardsSkeleton()
                                    .id(Self.skeletonID)
                                    .transition(GuideMotion.timelineInsertion(reduceMotion: reduceMotion))
                            }

                            if viewModel.isAssistantThinking {
                                AssistantThinkingRow()
                                    .id(Self.thinkingID)
                                    .transition(GuideMotion.timelineInsertion(reduceMotion: reduceMotion))
                            }

                            GeometryReader { geo in
                                Color.clear.preference(
                                    key: BottomAnchorOffsetKey.self,
                                    value: geo.frame(in: .named(Self.scrollSpace)).minY
                                )
                            }
                            .frame(height: 1)
                            .id(Self.bottomAnchorID)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 18)
                        .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: viewModel.timeline.map(\.id))
                        .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: viewModel.isAwaitingCards)
                        .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: viewModel.isAssistantThinking)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(GuideTheme.pageBackground)
                    .coordinateSpace(.named(Self.scrollSpace))
                    .background {
                        GeometryReader { geo in
                            Color.clear
                                .onAppear { viewportHeight = geo.size.height }
                                .onChange(of: geo.size.height) { _, height in
                                    viewportHeight = height
                                }
                        }
                    }
                    .onPreferenceChange(BottomAnchorOffsetKey.self) { minY in
                        isPinnedToBottom = minY <= viewportHeight + Self.pinThreshold
                    }
                    .dismissesKeyboardOnScroll()
                    .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
                    .onChange(of: viewModel.timeline) { _, _ in
                        // Streaming mutates the timeline on every token; only follow when the user is
                        // at the bottom, and without animation so it never fights a manual scroll.
                        guard isPinnedToBottom else { return }
                        proxy.scrollTo(Self.bottomAnchorID, anchor: .bottom)
                    }
                    .onChange(of: viewModel.isAwaitingCards) { _, awaiting in
                        guard awaiting, isPinnedToBottom else { return }
                        withAnimation(GuideMotion.scroll) {
                            proxy.scrollTo(Self.bottomAnchorID, anchor: .bottom)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                Divider()
                    .overlay(GuideTheme.line)

                ChatComposerView(
                    text: $viewModel.draftMessage,
                    isSending: viewModel.isSending,
                    isListening: viewModel.isListening,
                    cameraAction: cameraAction,
                    photoLibraryAction: photoLibraryAction,
                    voiceAction: { viewModel.toggleVoiceInput() },
                    sendAction: {
                        isPinnedToBottom = true
                        viewModel.sendDraftMessage()
                    },
                    cancelAction: { viewModel.cancelStreaming() }
                )
            }
            .background(GuideTheme.pageBackground)
            .keyboardDismissToolbar()
            .overlayPreferenceValue(GuideAnchorKey.self) { anchors in
                flightOverlay(anchors: anchors)
            }
            .sheet(item: $selectedProduct) { product in
                ProductDetailSheet(product: product) {
                    viewModel.addToCart(product: product)
                    selectedProduct = nil
                }
                .environmentObject(favourites)
            }
            .sheet(isPresented: $isCartPresented) {
                CartSheetView(
                    items: viewModel.cartItems,
                    quantityAction: { viewModel.updateCartItem(productID: $0, delta: $1) },
                    removeAction: { viewModel.removeFromCart(productID: $0) },
                    checkoutAction: checkoutAction
                )
            }
            .sheet(isPresented: $isFavouritesPresented) {
                FavouritesSheetView(
                    addToCartAction: { viewModel.addToCart(product: $0) }
                )
                .environmentObject(favourites)
            }
            .sheet(isPresented: $isSettingsPresented) {
                ChatSettingsSheetView(
                    isAutoReadingEnabled: Binding(
                        get: { viewModel.isAssistantSpeechEnabled },
                        set: { viewModel.setAssistantSpeechEnabled($0) }
                    )
                )
            }
        }
        .environmentObject(favourites)
    }

    /// Adds to the cart and, when motion is allowed, launches the flight thumbnail
    /// from the product card to the cart pill.
    private func addToCart(_ product: Product) {
        if !reduceMotion {
            flight = CartFlight(id: UUID(), product: product)
        }
        viewModel.addToCart(product: product)
    }

    @ViewBuilder
    private func flightOverlay(anchors: GuideAnchorKey.Value) -> some View {
        GeometryReader { proxy in
            if let flight,
               let source = anchors.productImages[flight.product.id],
               let target = anchors.cartPill {
                CartFlightView(
                    flight: flight,
                    from: CGPoint(x: proxy[source].midX, y: proxy[source].midY),
                    to: CGPoint(x: proxy[target].midX, y: proxy[target].midY),
                    onFinished: { self.flight = nil }
                )
                .id(flight.id)
            }
        }
        .allowsHitTesting(false)
    }
}

/// Carries the bottom anchor's vertical offset within the scroll viewport, used to decide whether
/// the user is at the bottom (and should keep following new content) or has scrolled up to read.
private struct BottomAnchorOffsetKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ChatTimelineItemView: View {
    let item: ChatTimelineItem
    let isOrderActionable: Bool
    @Binding var shippingAddress: String
    let retryAction: () -> Void
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void
    let orderReplyAction: (String) -> Void
    let speechPlaybackState: (ChatMessage) -> TextToSpeechPlaybackState?
    let speakAction: (ChatMessage) -> Void
    let stopSpeechAction: () -> Void

    var body: some View {
        switch item {
        case .message(let message):
            MessageBubbleView(
                message: message,
                speechPlaybackState: speechPlaybackState(message),
                speakAction: speakAction,
                stopSpeechAction: stopSpeechAction
            )
        case .plan(_, let steps):
            PlanStatusView(steps: steps)
        case .products(_, let products):
            ProductCarouselView(
                products: products,
                productAction: productAction,
                addToCartAction: addToCartAction
            )
        case .comparison(_, let comparison):
            ProductComparisonView(
                comparison: comparison,
                productAction: productAction,
                pickAction: addToCartAction
            )
        case .cartStatus(_, let text):
            CartStatusView(text: text)
        case .orderStatus(_, let order):
            OrderCardView(order: order, isActionable: isOrderActionable, shippingAddress: $shippingAddress, replyAction: orderReplyAction)
        case .error(_, let message):
            ErrorRetryView(message: message, retryAction: retryAction)
        }
    }
}

/// A standalone assistant bubble holding only the typing dots, shown during the bare gaps in a turn
/// (e.g. after the cards land while the narration is still being written).
@available(iOS 17.0, macOS 13.0, *)
private struct AssistantThinkingRow: View {
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            AssistantAvatarView()

            TypingIndicatorView()
                .padding(.horizontal, 13)
                .padding(.vertical, 10)
                .background(GuideTheme.assistantBubble)
                .clipShape(RoundedRectangle(cornerRadius: GuideTheme.bubbleRadius, style: .continuous))

            Spacer(minLength: 42)
        }
    }
}
