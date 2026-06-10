import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct FavouritesSheetView: View {
    let addToCartAction: (Product) -> Void

    @EnvironmentObject private var favourites: FavouritesStore
    @Environment(\.dismiss) private var dismiss
    @State private var openSwipeID: String?
    @State private var selectedProduct: Product?

    var body: some View {
        NavigationStack {
            Group {
                if favourites.items.isEmpty {
                    emptyState
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        VStack(spacing: 0) {
                            ForEach(favourites.items) { product in
                                FavouriteRowView(
                                    product: product,
                                    productAction: { selectedProduct = product },
                                    addToCartAction: { addToCartAction(product) }
                                )
                                .guideSwipeActions(itemID: product.id, openItemID: $openSwipeID, actions: [
                                    SwipeAction(systemImage: "trash", title: "移除", tint: GuideTheme.warning) {
                                        withAnimation(GuideMotion.snappy) {
                                            favourites.remove(productID: product.id)
                                        }
                                    }
                                ])
                                .transition(.opacity.combined(with: .scale(scale: 0.96)))
                            }
                        }
                        .animation(GuideMotion.snappy, value: favourites.items.map(\.id))
                        .padding(.horizontal, 16)
                    }
                }
            }
            .background(GuideTheme.pageBackground)
            .navigationTitle("我的收藏")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") {
                        dismiss()
                    }
                    .foregroundStyle(GuideTheme.secondaryInk)
                }
            }
            .sheet(item: $selectedProduct) { product in
                ProductDetailSheet(product: product) {
                    addToCartAction(product)
                    selectedProduct = nil
                }
                .environmentObject(favourites)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "heart")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(GuideTheme.tertiaryInk)
                .frame(width: 72, height: 72)
                .background(GuideTheme.assistantBubble)
                .clipShape(Circle())

            Text("还没有收藏的商品")
                .font(.headline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)

            Text("点击商品卡片上的爱心，把心仪的商品收藏到这里。")
                .font(.subheadline)
                .foregroundStyle(GuideTheme.secondaryInk)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 28)
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
private struct FavouriteRowView: View {
    let product: Product
    let productAction: () -> Void
    let addToCartAction: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Button(action: productAction) {
                HStack(spacing: 10) {
                    ProductImageView(product: product)
                        .frame(width: 52, height: 52)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                    VStack(alignment: .leading, spacing: 5) {
                        Text(product.title)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(GuideTheme.inkStrong)
                            .lineLimit(1)
                            .minimumScaleFactor(0.82)

                        ProductSpecLine(product: product)

                        Text(product.formattedPrice)
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(GuideTheme.accent)
                    }
                }
            }
            .buttonStyle(.plain)

            Spacer(minLength: 8)

            Button(action: addToCartAction) {
                Label("加购", systemImage: "plus")
                    .labelStyle(.titleAndIcon)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 6)
                    .background(GuideTheme.accent)
                    .clipShape(Capsule())
            }
            .buttonStyle(PressableButtonStyle())
            .accessibilityLabel("将 \(product.title) 加入购物车")
        }
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(GuideTheme.line)
                .frame(height: 1)
        }
    }
}
