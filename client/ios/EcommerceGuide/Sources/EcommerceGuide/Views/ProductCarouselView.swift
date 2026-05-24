import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct ProductCarouselView: View {
    let products: [Product]
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text("Recommended products")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(GuideTheme.ink)

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 12) {
                    ForEach(products) { product in
                        ProductCardView(
                            product: product,
                            productAction: productAction,
                            addToCartAction: addToCartAction
                        )
                    }
                }
                .padding(.trailing, 16)
            }
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct ProductCardView: View {
    let product: Product
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Button {
                productAction(product)
            } label: {
                VStack(alignment: .leading, spacing: 9) {
                    ProductImageView(product: product)
                        .frame(width: 210, height: 132)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                    VStack(alignment: .leading, spacing: 4) {
                        Text(product.brand)
                            .font(.caption.weight(.medium))
                            .foregroundStyle(GuideTheme.secondaryInk)
                            .lineLimit(1)

                        Text(product.title)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(GuideTheme.ink)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)

                        if let reason = product.reason, !reason.isEmpty {
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .lineLimit(2)
                                .multilineTextAlignment(.leading)
                        }

                        Text(product.formattedPrice)
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(GuideTheme.accent)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .buttonStyle(.plain)
            .accessibilityHint("Opens product details")

            Button {
                addToCartAction(product)
            } label: {
                Label("Add", systemImage: "plus")
                    .font(.caption.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: 44)
            }
            .buttonStyle(.borderedProminent)
            .tint(GuideTheme.accent)
            .accessibilityLabel("Add \(product.title) to cart")
        }
        .padding(10)
        .frame(width: 230, alignment: .leading)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(GuideTheme.line)
        }
        .accessibilityLabel("\(product.title), \(product.formattedPrice)")
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct ProductImageView: View {
    let product: Product

    var body: some View {
        if let url = imageURL {
            AsyncImage(url: url) { phase in
                switch phase {
                case .success(let image):
                    image
                        .resizable()
                        .scaledToFill()
                case .failure:
                    placeholder
                case .empty:
                    ZStack {
                        placeholder
                        ProgressView()
                            .tint(GuideTheme.accent)
                    }
                @unknown default:
                    placeholder
                }
            }
        } else {
            placeholder
        }
    }

    private var imageURL: URL? {
        if let url = URL(string: product.imagePath), url.scheme != nil {
            return url
        }

        return productImageBaseURL
            .appending(path: "assets")
            .appending(path: "products")
            .appending(path: product.imagePath)
    }

    private var productImageBaseURL: URL {
        let configured = UserDefaults.standard.string(forKey: "EcommerceGuideBackendURL")
            .flatMap(URL.init(string:))
        let endpoint = configured ?? URL(string: "http://127.0.0.1:8000/api/v1/chat/stream")!

        var components = URLComponents()
        components.scheme = endpoint.scheme
        components.host = endpoint.host
        components.port = endpoint.port
        return components.url ?? URL(string: "http://127.0.0.1:8000")!
    }

    private var placeholder: some View {
        ZStack {
            Rectangle()
                .fill(GuideTheme.accentSoft)

            VStack(spacing: 7) {
                Image(systemName: "shippingbox")
                    .font(.title2)
                    .foregroundStyle(GuideTheme.accent)

                Text(product.category)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .lineLimit(1)
            }
            .padding(12)
        }
    }
}
