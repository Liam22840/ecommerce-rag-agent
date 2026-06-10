import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductDetailSheet: View {
    let product: Product
    let addToCartAction: () -> Void

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var favourites: FavouritesStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ProductMediaPager(product: product)
                        .frame(maxWidth: .infinity)
                        .frame(height: 300)
                        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                                .stroke(GuideTheme.line)
                        }
                        .shadow(color: GuideTheme.cardShadow, radius: 6, y: 2)

                    VStack(alignment: .leading, spacing: 9) {
                        Text(product.brand)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(GuideTheme.tertiaryInk)
                            .textCase(.uppercase)

                        Text(product.title)
                            .font(.title2.weight(.bold))
                            .foregroundStyle(GuideTheme.inkStrong)
                            .fixedSize(horizontal: false, vertical: true)

                        HStack(alignment: .center) {
                            Text(product.formattedPrice)
                                .font(.title3.weight(.bold))
                                .foregroundStyle(GuideTheme.accent)

                            Spacer()

                            FavouriteButton(isFavourite: favourites.isFavourite(product)) {
                                withAnimation(GuideMotion.snappy) {
                                    favourites.toggle(product)
                                }
                            }
                        }

                        if let priceSummary = product.priceSummary,
                           !priceSummary.isEmpty,
                           priceSummary != product.formattedPrice {
                            Text(priceSummary)
                                .font(.footnote)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    HStack(spacing: 8) {
                        Label(product.category, systemImage: "tag")
                        Text(product.subCategory)
                    }
                    .font(.caption.weight(.medium))
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .background(GuideTheme.assistantBubble)
                    .clipShape(Capsule())

                    if let reason = product.reason, !reason.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("推荐理由")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(GuideTheme.inkStrong)

                            Text(reason)
                                .font(.subheadline)
                                .lineSpacing(2)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding(14)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(GuideTheme.panelBackground)
                        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                                .stroke(GuideTheme.line)
                        }
                    }
                }
                .padding(16)
            }
            .background(GuideTheme.pageBackground)
            .safeAreaInset(edge: .bottom) {
                Button(action: addToCartAction) {
                    Label("加入购物车", systemImage: "cart.badge.plus")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .frame(height: 46)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .background(GuideTheme.accent)
                .clipShape(Capsule())
                .shadow(color: GuideTheme.accentShadow, radius: 8, y: 3)
                .padding(16)
                .background(GuideTheme.panelBackground)
            }
            .navigationTitle("商品详情")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                    }
                    .accessibilityLabel("关闭")
                }
            }
            .presentationDragIndicator(.visible)
        }
    }
}
