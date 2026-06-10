import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
public struct ChatScreen: View {
    @StateObject private var viewModel: ChatViewModel
    @StateObject private var favourites = FavouritesStore()
    @State private var selectedProduct: Product?
    @State private var isCartPresented = false
    @State private var isFavouritesPresented = false
    @State private var flight: CartFlight?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let cameraAction: () -> Void
    private let checkoutAction: () -> Void

    private static let skeletonID = "products-skeleton"

    @MainActor
    public init() {
        _viewModel = StateObject(wrappedValue: ChatViewModel())
        self.cameraAction = {}
        self.checkoutAction = {}
    }

    @MainActor
    public init(
        viewModel: ChatViewModel,
        cameraAction: @escaping () -> Void = {},
        checkoutAction: @escaping () -> Void = {}
    ) {
        _viewModel = StateObject(wrappedValue: viewModel)
        self.cameraAction = cameraAction
        self.checkoutAction = checkoutAction
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ChatHeaderView(
                    cartItems: viewModel.cartItems,
                    favouritesCount: favourites.items.count,
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
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 18)
                        .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: viewModel.timeline.map(\.id))
                        .animation(GuideMotion.entrance(reduceMotion: reduceMotion), value: viewModel.isAwaitingCards)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(GuideTheme.pageBackground)
                    .dismissesKeyboardOnScroll()
                    .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
                    .onChange(of: viewModel.timeline) { _, items in
                        guard let id = items.last?.id else {
                            return
                        }

                        withAnimation(GuideMotion.scroll) {
                            proxy.scrollTo(id, anchor: .bottom)
                        }
                    }
                    .onChange(of: viewModel.isAwaitingCards) { _, awaiting in
                        guard awaiting else {
                            return
                        }

                        withAnimation(GuideMotion.scroll) {
                            proxy.scrollTo(Self.skeletonID, anchor: .bottom)
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
                    voiceAction: { viewModel.toggleVoiceInput() },
                    sendAction: { viewModel.sendDraftMessage() },
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

@available(iOS 17.0, macOS 13.0, *)
private struct ChatTimelineItemView: View {
    let item: ChatTimelineItem
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
            OrderCardView(order: order, shippingAddress: $shippingAddress, replyAction: orderReplyAction)
        case .error(_, let message):
            ErrorRetryView(message: message, retryAction: retryAction)
        }
    }
}
