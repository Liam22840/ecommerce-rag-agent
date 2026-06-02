import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ChatHeaderView: View {
    let cartItems: [CartItem]
    let cartAction: () -> Void

    var body: some View {
        HStack(alignment: .center) {
            Color.clear
                .frame(width: 48, height: 36)

            Spacer(minLength: 8)

            Text("AI 购物助手")
                .font(.headline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)
                .lineLimit(1)
                .minimumScaleFactor(0.78)

            Spacer(minLength: 8)

            CartPillView(cartItems: cartItems, action: cartAction)
        }
        .frame(minHeight: 44)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(GuideTheme.panelBackground)
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct CartPillView: View {
    let cartItems: [CartItem]
    let action: () -> Void

    private var itemCount: Int {
        cartItems.reduce(0) { $0 + $1.quantity }
    }

    private var total: Decimal {
        cartItems.reduce(Decimal.zero) { partialResult, item in
            partialResult + (item.product.basePrice * Decimal(item.quantity))
        }
    }

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "bag")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(GuideTheme.inkStrong)
                    .frame(width: 36, height: 36)

                if itemCount > 0 {
                    Text("\(itemCount)")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(minWidth: 16, minHeight: 16)
                        .padding(.horizontal, itemCount > 9 ? 3 : 0)
                        .background(GuideTheme.accent)
                        .clipShape(Capsule())
                        .offset(x: 4, y: -1)
                }
            }
            .frame(width: 48, height: 36, alignment: .trailing)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("购物车，\(itemCount) 件商品，\(formattedTotal)")
    }

    private var formattedTotal: String {
        let value = NSDecimalNumber(decimal: total)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(total)"
    }
}
