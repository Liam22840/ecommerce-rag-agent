import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductComparisonView: View {
    let products: [Product]
    let productAction: (Product) -> Void
    let pickAction: (Product) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            AssistantAvatarView()

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 8) {
                    ForEach(products) { product in
                        ProductComparisonCard(
                            product: product,
                            productAction: productAction,
                            pickAction: pickAction
                        )
                    }
                }
                .padding(.bottom, 4)
            }
            .frame(maxWidth: 610, alignment: .leading)

            Spacer(minLength: 42)
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ProductComparisonCard: View {
    let product: Product
    let productAction: (Product) -> Void
    let pickAction: (Product) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Button {
                productAction(product)
            } label: {
                ProductImageView(product: product)
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            .buttonStyle(.plain)
            .accessibilityHint("打开商品详情")

            Button {
                productAction(product)
            } label: {
                VStack(alignment: .leading, spacing: 4) {
                    Text(product.title)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(GuideTheme.inkStrong)
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
            .buttonStyle(.plain)
            .padding(.top, 2)
        }
        .padding(10)
        .frame(width: 180, alignment: .topLeading)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(Color.black.opacity(0.04))
        }
        .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(product.title), \(product.formattedPrice)")
    }
}
