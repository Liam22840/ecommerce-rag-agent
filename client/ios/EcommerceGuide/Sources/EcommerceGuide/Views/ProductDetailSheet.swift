import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductDetailSheet: View {
    let product: Product
    let addToCartAction: () -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    ProductImageView(product: product)
                        .frame(maxWidth: .infinity)
                        .frame(height: 280)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                    VStack(alignment: .leading, spacing: 8) {
                        Text(product.brand)
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(GuideTheme.secondaryInk)

                        Text(product.title)
                            .font(.title2.weight(.bold))
                            .foregroundStyle(GuideTheme.ink)
                            .fixedSize(horizontal: false, vertical: true)

                        Text(product.formattedPrice)
                            .font(.title3.weight(.bold))
                            .foregroundStyle(GuideTheme.accent)
                    }

                    HStack(spacing: 8) {
                        Label(product.category, systemImage: "tag")
                        Text(product.subCategory)
                    }
                    .font(.caption.weight(.medium))
                    .foregroundStyle(GuideTheme.secondaryInk)

                    if let reason = product.reason, !reason.isEmpty {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("Why it fits")
                                .font(.headline)
                                .foregroundStyle(GuideTheme.ink)

                            Text(reason)
                                .font(.body)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(16)
            }
            .background(GuideTheme.pageBackground)
            .safeAreaInset(edge: .bottom) {
                Button(action: addToCartAction) {
                    Label("Add to cart", systemImage: "cart.badge.plus")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .frame(height: 46)
                }
                .buttonStyle(.borderedProminent)
                .tint(GuideTheme.accent)
                .padding(16)
                .background(.regularMaterial)
            }
            .navigationTitle("Product details")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                    }
                    .accessibilityLabel("Close")
                }
            }
        }
    }
}
