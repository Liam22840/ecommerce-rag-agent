import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
public struct ChatScreen: View {
    @StateObject private var viewModel: ChatViewModel
    @State private var selectedProduct: Product?

    @MainActor
    public init() {
        _viewModel = StateObject(wrappedValue: ChatViewModel())
    }

    @MainActor
    public init(viewModel: ChatViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ChatHeaderView(cartItems: viewModel.cartItems)

                Divider()

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            ForEach(viewModel.timeline) { item in
                                ChatTimelineItemView(
                                    item: item,
                                    retryAction: { viewModel.retryLastMessage() },
                                    productAction: { selectedProduct = $0 },
                                    addToCartAction: { viewModel.addToCart(product: $0) }
                                )
                                .id(item.id)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 18)
                    }
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

                Divider()

                ChatComposerView(
                    text: $viewModel.draftMessage,
                    isSending: viewModel.isSending,
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
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ChatTimelineItemView: View {
    let item: ChatTimelineItem
    let retryAction: () -> Void
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void

    var body: some View {
        switch item {
        case .message(let message):
            MessageBubbleView(message: message)
        case .products(_, let products):
            ProductCarouselView(
                products: products,
                productAction: productAction,
                addToCartAction: addToCartAction
            )
        case .cartStatus(_, let text):
            CartStatusView(text: text)
        case .error(_, let message):
            ErrorRetryView(message: message, retryAction: retryAction)
        }
    }
}
