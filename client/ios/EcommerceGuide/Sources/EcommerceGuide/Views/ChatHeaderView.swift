import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ChatHeaderView: View {
    let cartItems: [CartItem]
    let favouritesCount: Int
    let settingsAction: () -> Void
    let cartAction: () -> Void
    let favouritesAction: () -> Void

    var body: some View {
        ZStack {
            Text("AI 购物助手")
                .font(.headline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)
                .lineLimit(1)
                .minimumScaleFactor(0.78)
                .padding(.horizontal, 112)

            HStack(alignment: .center, spacing: 8) {
                FavouritesPillView(count: favouritesCount, action: favouritesAction)

                Spacer(minLength: 8)

                SettingsButton(action: settingsAction)
                CartPillView(cartItems: cartItems, action: cartAction)
            }
        }
        .frame(minHeight: 44)
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(GuideTheme.panelBackground)
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct SettingsButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "gearshape")
                .font(.system(size: 19, weight: .semibold))
                .foregroundStyle(GuideTheme.inkStrong)
                .frame(width: 36, height: 36)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("设置")
        .help("设置")
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct FavouritesPillView: View {
    let count: Int
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "heart")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(GuideTheme.inkStrong)
                    .frame(width: 36, height: 36)
                    .symbolEffect(.bounce, value: count)

                if count > 0 {
                    Text("\(count)")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(minWidth: 16, minHeight: 16)
                        .padding(.horizontal, count > 9 ? 3 : 0)
                        .background(GuideTheme.favourite)
                        .clipShape(Capsule())
                        .contentTransition(.numericText(value: Double(count)))
                        .transition(.scale.combined(with: .opacity))
                        .offset(x: 4, y: -1)
                }
            }
            .animation(GuideMotion.snappy, value: count)
            .frame(width: 48, height: 36, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("我的收藏，\(count) 件商品")
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
                    .symbolEffect(.bounce, value: itemCount)

                if itemCount > 0 {
                    Text("\(itemCount)")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(minWidth: 16, minHeight: 16)
                        .padding(.horizontal, itemCount > 9 ? 3 : 0)
                        .background(GuideTheme.accent)
                        .clipShape(Capsule())
                        .contentTransition(.numericText(value: Double(itemCount)))
                        .transition(.scale.combined(with: .opacity))
                        .offset(x: 4, y: -1)
                }
            }
            .animation(GuideMotion.snappy, value: itemCount)
            .frame(width: 48, height: 36, alignment: .trailing)
        }
        .buttonStyle(.plain)
        .anchorPreference(key: GuideAnchorKey.self, value: .bounds) { anchor in
            GuideAnchorKey.Value(cartPill: anchor)
        }
        .sensoryFeedback(.impact(weight: .light), trigger: itemCount)
        .accessibilityLabel("购物车，\(itemCount) 件商品，\(formattedTotal)")
    }

    private var formattedTotal: String {
        let value = NSDecimalNumber(decimal: total)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(total)"
    }
}
