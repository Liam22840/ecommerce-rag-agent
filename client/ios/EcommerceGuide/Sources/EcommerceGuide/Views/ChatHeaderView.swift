import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct ChatHeaderView: View {
    let cartItems: [CartItem]

    var body: some View {
        ViewThatFits(in: .horizontal) {
            horizontalLayout
            verticalLayout
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(GuideTheme.panelBackground)
    }

    private var horizontalLayout: some View {
        HStack(alignment: .center, spacing: 12) {
            titleBlock

            Spacer(minLength: 10)

            CartPillView(cartItems: cartItems)
        }
    }

    private var verticalLayout: some View {
        VStack(alignment: .leading, spacing: 10) {
            titleBlock
            CartPillView(cartItems: cartItems)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Ecommerce Guide")
                .font(.headline.weight(.semibold))
                .foregroundStyle(GuideTheme.ink)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text("Catalog search and shopping assistant")
                .font(.caption)
                .foregroundStyle(GuideTheme.secondaryInk)
                .lineLimit(2)
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct CartPillView: View {
    let cartItems: [CartItem]

    private var itemCount: Int {
        cartItems.reduce(0) { $0 + $1.quantity }
    }

    private var total: Decimal {
        cartItems.reduce(Decimal.zero) { partialResult, item in
            partialResult + (item.product.basePrice * Decimal(item.quantity))
        }
    }

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: "cart")
                .font(.system(size: 14, weight: .semibold))

            Text("\(itemCount)")
                .font(.subheadline.weight(.semibold))

            if itemCount > 0 {
                Text(formattedTotal)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
        }
        .foregroundStyle(GuideTheme.ink)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(minHeight: 44)
        .background(GuideTheme.accentSoft)
        .clipShape(Capsule())
        .accessibilityLabel("Cart, \(itemCount) items, \(formattedTotal)")
    }

    private var formattedTotal: String {
        let value = NSDecimalNumber(decimal: total)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(total)"
    }
}
