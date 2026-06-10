import SwiftUI

/// Swipeable media pager for the product detail sheet. The dataset ships one photo per
/// product, so the extra pages carry real content (推荐理由 / 规格) instead of repeated images.
@available(iOS 17.0, macOS 14.0, *)
struct ProductMediaPager: View {
    let product: Product

    @State private var page = 0
    @State private var zoom: CGFloat = 1

    var body: some View {
        TabView(selection: $page) {
            ZoomableProductImage(product: product, zoom: $zoom).tag(0)

            if let reason = product.reason, !reason.isEmpty {
                infoPage(title: "推荐理由", systemImage: "sparkles") {
                    Text(reason)
                        .font(.subheadline)
                        .lineSpacing(3)
                        .foregroundStyle(GuideTheme.secondaryInk)
                }.tag(1)
            }

            if hasSpecPage {
                infoPage(title: "规格信息", systemImage: "list.bullet.rectangle") {
                    VStack(alignment: .leading, spacing: 8) {
                        if let spec = product.spec, !spec.isEmpty {
                            specRow(label: "规格", value: spec)
                        }
                        if let rating = product.rating {
                            specRow(label: "评分", value: String(format: "%.1f / 5", rating))
                        }
                        if let sales = product.sales, !sales.isEmpty {
                            specRow(label: "月销", value: sales)
                        }
                    }
                }.tag(2)
            }
        }
        #if os(iOS)
        .tabViewStyle(.page(indexDisplayMode: .automatic))
        .indexViewStyle(.page(backgroundDisplayMode: .interactive))
        #endif
        .onChange(of: page) { _, _ in
            withAnimation(GuideMotion.snappy) {
                zoom = 1
            }
        }
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
    }

    private var hasSpecPage: Bool {
        (product.spec?.isEmpty == false) || product.rating != nil || (product.sales?.isEmpty == false)
    }

    private func infoPage(
        title: String,
        systemImage: String,
        @ViewBuilder content: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)

            content()

            Spacer(minLength: 0)
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(GuideTheme.accentSoft.opacity(0.5))
    }

    private func specRow(label: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(GuideTheme.tertiaryInk)
                .frame(width: 36, alignment: .leading)

            Text(value)
                .font(.subheadline)
                .foregroundStyle(GuideTheme.secondaryInk)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
private struct ZoomableProductImage: View {
    let product: Product

    @Binding var zoom: CGFloat
    @GestureState private var magnify: CGFloat = 1

    var body: some View {
        ProductImageView(product: product)
            .scaleEffect(zoom * magnify)
            .gesture(
                MagnifyGesture()
                    .updating($magnify) { value, state, _ in
                        state = value.magnification
                    }
                    .onEnded { value in
                        zoom = min(3, max(1, zoom * value.magnification))
                    }
            )
            .onTapGesture(count: 2) {
                withAnimation(GuideMotion.snappy) {
                    zoom = 1
                }
            }
            .clipped()
            .accessibilityLabel("商品图片，双指缩放查看")
    }
}
