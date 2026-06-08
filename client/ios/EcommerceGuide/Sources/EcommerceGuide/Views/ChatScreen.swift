import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
public struct ChatScreen: View {
    @StateObject private var viewModel: ChatViewModel
    @State private var selectedProduct: Product?
    @State private var isCartPresented = false
    private let cameraAction: () -> Void
    private let checkoutAction: () -> Void

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
                    cartAction: { isCartPresented = true }
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
                                    retryAction: { viewModel.retryLastMessage() },
                                    productAction: { selectedProduct = $0 },
                                    addToCartAction: { viewModel.addToCart(product: $0) },
                                    orderReplyAction: { viewModel.sendQuickReply($0) }
                                )
                                .id(item.id)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 18)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(GuideTheme.pageBackground)
                    .onChange(of: viewModel.timeline) { items in
                        guard let id = items.last?.id else {
                            return
                        }

                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(id, anchor: .bottom)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                Divider()
                    .overlay(GuideTheme.line)

                ChatComposerView(
                    text: $viewModel.draftMessage,
                    isSending: viewModel.isSending,
                    cameraAction: cameraAction,
                    sendAction: { viewModel.sendDraftMessage() },
                    cancelAction: { viewModel.cancelStreaming() }
                )
            }
            .background(GuideTheme.pageBackground)
            .sheet(item: $selectedProduct) { product in
                ProductDetailSheet(product: product) {
                    viewModel.addToCart(product: product)
                    selectedProduct = nil
                }
            }
            .sheet(isPresented: $isCartPresented) {
                CartSheetView(
                    items: viewModel.cartItems,
                    quantityAction: { viewModel.updateCartItem(productID: $0, delta: $1) },
                    removeAction: { viewModel.removeFromCart(productID: $0) },
                    checkoutAction: checkoutAction
                )
            }
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ChatTimelineItemView: View {
    let item: ChatTimelineItem
    let retryAction: () -> Void
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void
    let orderReplyAction: (String) -> Void

    var body: some View {
        switch item {
        case .message(let message):
            MessageBubbleView(message: message)
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
            OrderCardView(order: order, replyAction: orderReplyAction)
        case .error(_, let message):
            ErrorRetryView(message: message, retryAction: retryAction)
        }
    }
}
