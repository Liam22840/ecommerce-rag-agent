import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductComparisonView: View {
    let comparison: ProductComparison
    let productAction: (Product) -> Void
    let pickAction: (Product) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            AssistantAvatarView()

            VStack(alignment: .leading, spacing: 10) {
                if let summary = comparison.summary, !summary.isEmpty {
                    Text(summary)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(GuideTheme.inkStrong)
                        .fixedSize(horizontal: false, vertical: true)
                }

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 8) {
                        ForEach(comparison.products) { product in
                            ProductComparisonCard(
                                product: product,
                                isWinner: product.id == comparison.winnerProductID,
                                productAction: productAction,
                                pickAction: pickAction
                            )
                            .scrollTransition { content, phase in
                                content
                                    .scaleEffect(phase.isIdentity ? 1 : 0.95)
                                    .opacity(phase.isIdentity ? 1 : 0.75)
                            }
                        }
                    }
                    .scrollTargetLayout()
                    .padding(.bottom, 4)
                }
                .scrollTargetBehavior(.viewAligned)

                if !comparison.rows.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(comparison.rows, id: \.dimension) { row in
                            ComparisonDimensionRow(row: row, products: comparison.products)
                        }
                    }
                }

                if let recommendation = comparison.recommendation, !recommendation.isEmpty {
                    Text(recommendation)
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(GuideTheme.accent)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.top, 2)
                }
            }
            .padding(12)
            .frame(maxWidth: 650, alignment: .leading)
            .background(GuideTheme.panelBackground)
            .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                    .stroke(Color.black.opacity(0.04))
            }
            .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)

            Spacer(minLength: 42)
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ProductComparisonCard: View {
    let product: Product
    let isWinner: Bool
    let productAction: (Product) -> Void
    let pickAction: (Product) -> Void

    @EnvironmentObject private var favourites: FavouritesStore

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Button {
                productAction(product)
            } label: {
                ProductImageView(product: product)
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            .buttonStyle(PressableButtonStyle())
            .overlay(alignment: .topTrailing) {
                FavouriteButton(isFavourite: favourites.isFavourite(product), compact: true) {
                    withAnimation(GuideMotion.snappy) {
                        favourites.toggle(product)
                    }
                }
                .offset(x: 5, y: -5)
            }
            .accessibilityHint("打开商品详情")

            Button {
                productAction(product)
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(product.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(isWinner ? GuideTheme.accent : GuideTheme.inkStrong)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)

                    ProductSpecLine(product: product)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)

            Text(product.formattedPrice)
                .font(.subheadline.weight(.bold))
                .foregroundStyle(GuideTheme.accent)
                .lineLimit(1)
                .minimumScaleFactor(0.75)

            ProductRatingSalesRow(product: product)

            ProductProsConsView(product: product)

            if isWinner {
                Label("推荐", systemImage: "checkmark.seal.fill")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(GuideTheme.accent)
                    .lineLimit(1)
            }

            Button {
                pickAction(product)
            } label: {
                Text("选这个")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 32)
                    .background(GuideTheme.accent)
                    .clipShape(Capsule())
            }
            .buttonStyle(PressableButtonStyle())
            .padding(.top, 2)
        }
        .padding(10)
        .frame(width: 180, alignment: .topLeading)
        .background(isWinner ? GuideTheme.accentSoft : GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(isWinner ? GuideTheme.accent.opacity(0.32) : Color.black.opacity(0.04))
        }
        .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(product.title), \(product.formattedPrice)")
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ComparisonDimensionRow: View {
    let row: ComparisonRow
    let products: [Product]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(row.dimension)
                    .font(.caption.weight(.bold))
                    .foregroundStyle(GuideTheme.inkStrong)

                if winnerProduct != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(GuideTheme.accent)
                }
            }

            ForEach(row.values, id: \.productID) { value in
                HStack(alignment: .top, spacing: 6) {
                    Text(shortName(for: value.productID))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(value.productID == row.winnerProductID ? GuideTheme.accent : GuideTheme.secondaryInk)
                        .frame(width: 58, alignment: .leading)
                        .lineLimit(1)

                    Text(value.value)
                        .font(.caption2)
                        .foregroundStyle(GuideTheme.secondaryInk)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Text(row.verdict)
                .font(.caption2)
                .foregroundStyle(GuideTheme.tertiaryInk)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 1)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(GuideTheme.assistantBubble)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var winnerProduct: Product? {
        guard let productID = row.winnerProductID else {
            return nil
        }
        return products.first { $0.id == productID }
    }

    private func shortName(for productID: String) -> String {
        guard let product = products.first(where: { $0.id == productID }) else {
            return productID
        }
        return product.brand
    }
}
