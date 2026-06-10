import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductCarouselView: View {
    let products: [Product]
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void

    @State private var openSwipeID: String?

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            AssistantAvatarView()

            VStack(spacing: 6) {
                ForEach(products) { product in
                    ProductCardView(
                        product: product,
                        openSwipeID: $openSwipeID,
                        productAction: productAction,
                        addToCartAction: addToCartAction
                    )
                }
            }
            .frame(maxWidth: 610, alignment: .leading)

            Spacer(minLength: 42)
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct ProductCardView: View {
    let product: Product
    @Binding var openSwipeID: String?
    let productAction: (Product) -> Void
    let addToCartAction: (Product) -> Void

    @EnvironmentObject private var favourites: FavouritesStore

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Button {
                productAction(product)
            } label: {
                ProductImageView(product: product)
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            .buttonStyle(PressableButtonStyle())
            .anchorPreference(key: GuideAnchorKey.self, value: .bounds) { anchor in
                GuideAnchorKey.Value(productImages: [product.id: anchor])
            }
            .overlay(alignment: .topTrailing) {
                FavouriteButton(isFavourite: favourites.isFavourite(product), compact: true) {
                    withAnimation(GuideMotion.snappy) {
                        favourites.toggle(product)
                    }
                }
                .offset(x: 5, y: -5)
            }
            .accessibilityHint("打开商品详情")

            VStack(alignment: .leading, spacing: 3) {
                Button {
                    productAction(product)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(product.title)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(GuideTheme.inkStrong)
                            .lineLimit(1)
                            .minimumScaleFactor(0.82)
                            .multilineTextAlignment(.leading)

                        ProductSpecLine(product: product)

                        if let reason = product.reason, !reason.isEmpty {
                            Text(reason)
                                .font(.caption)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .lineLimit(1)
                                .multilineTextAlignment(.leading)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)

                ProductRatingSalesRow(product: product)

                HStack(alignment: .center, spacing: 8) {
                    Text(product.formattedPrice)
                        .font(.headline.weight(.bold))
                        .foregroundStyle(GuideTheme.accent)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                    Spacer(minLength: 8)

                    Button {
                        addToCartAction(product)
                    } label: {
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
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(Color.black.opacity(0.04))
        }
        .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)
        .guideSwipeActions(itemID: product.id, openItemID: $openSwipeID, actions: [
            SwipeAction(systemImage: "heart.fill", title: "收藏", tint: GuideTheme.favourite) {
                withAnimation(GuideMotion.snappy) {
                    favourites.toggle(product)
                }
            },
            SwipeAction(systemImage: "cart.badge.plus", title: "加购", tint: GuideTheme.accent) {
                addToCartAction(product)
            }
        ])
        .accessibilityLabel("\(product.title), \(product.formattedPrice)")
    }
}

@available(iOS 17.0, macOS 13.0, *)
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
                    SkeletonBlock(cornerRadius: 0)
                        .shimmer()
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
        let configured = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_BACKEND_URL"]
            ?? UserDefaults.standard.string(forKey: "EcommerceGuideBackendURL")
                .flatMap { $0.contains("192.168.0.184") ? nil : $0 }
        let configuredURL = configured
            .flatMap(URL.init(string:))
        let endpoint = configuredURL ?? URL(string: "http://192.168.0.176:8000/api/chat/stream")!

        var components = URLComponents()
        components.scheme = endpoint.scheme
        components.host = endpoint.host
        components.port = endpoint.port
        return components.url ?? URL(string: "http://192.168.0.176:8000")!
    }

    private var placeholder: some View {
        ZStack {
            LinearGradient(
                colors: [
                    ProductVisuals.softColor(for: product).opacity(0.9),
                    ProductVisuals.softColor(for: product).opacity(0.55)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            VStack(spacing: 7) {
                Image(systemName: ProductVisuals.symbol(for: product))
                    .font(.title2)
                    .foregroundStyle(GuideTheme.accent)

                Text(product.category)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .lineLimit(1)
            }
            .padding(12)
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private enum ProductVisuals {
    static func symbol(for product: Product) -> String {
        switch product.category.lowercased() {
        case let value where value.contains("apparel") || value.contains("服"):
            return "tshirt.fill"
        case let value where value.contains("footwear") || value.contains("鞋"):
            return "shoe.2.fill"
        case let value where value.contains("access") || value.contains("配饰") || value.contains("包"):
            return "bag.fill"
        case let value where value.contains("home") || value.contains("家居"):
            return "cup.and.saucer.fill"
        default:
            return "shippingbox.fill"
        }
    }

    static func softColor(for product: Product) -> Color {
        switch abs(product.id.hashValue) % 4 {
        case 0:
            return Color(red: 0.953, green: 0.878, blue: 0.827)
        case 1:
            return Color(red: 0.894, green: 0.929, blue: 0.996)
        case 2:
            return Color(red: 0.902, green: 0.949, blue: 0.902)
        default:
            return Color(red: 0.957, green: 0.918, blue: 0.976)
        }
    }
}
