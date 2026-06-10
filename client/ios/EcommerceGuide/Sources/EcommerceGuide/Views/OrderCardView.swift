import SwiftUI

/// Renders the backend's order state (待确认 / 已提交 / 已取消) as a card. When the order is awaiting
/// confirmation it offers 确认 / 取消 buttons that send the reply on the user's behalf, so the whole
/// 下单确认流程 is driven by the server's order state rather than plain text.
@available(iOS 17.0, macOS 13.0, *)
struct OrderCardView: View {
    let order: Order
    @Binding var shippingAddress: String
    let replyAction: (String) -> Void

    @State private var badgeAppeared = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusBadge

            if !order.summary.isEmpty {
                Text(order.summary)
                    .font(.footnote)
                    .foregroundStyle(GuideTheme.secondaryInk)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if order.isAwaitingConfirmation {
                // The shopper confirms or edits the address right here; 确认 sends it with the reply.
                // Editing in chat ("把地址改成…") flows the same way and re-syncs this field.
                EditableOrderField(title: "收货地址", text: $shippingAddress, axis: .vertical)

                HStack(spacing: 10) {
                    Button {
                        replyAction("取消订单")
                    } label: {
                        Text("取消")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(GuideTheme.secondaryInk)
                            .frame(maxWidth: .infinity)
                            .frame(height: 40)
                            .background(GuideTheme.pageBackground, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    .buttonStyle(.plain)

                    Button {
                        replyAction("确认")
                    } label: {
                        Text("确认下单")
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 40)
                            .background(GuideTheme.accent, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 2)
            }
        }
        .padding(14)
        .frame(maxWidth: 320, alignment: .leading)
        .background(GuideTheme.assistantBubble)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private var statusBadge: some View {
        let style = statusStyle
        return HStack(spacing: 8) {
            Image(systemName: style.icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(style.tint)
            Text(style.label)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(style.tint)
        }
        .scaleEffect(badgeAppeared ? 1 : 0.85)
        .opacity(badgeAppeared ? 1 : 0)
        .animation(GuideMotion.snappy, value: badgeAppeared)
        .onAppear {
            badgeAppeared = true
        }
    }

    private var statusStyle: (label: String, icon: String, tint: Color) {
        switch order.status {
        case "submitted": return ("订单已提交", "checkmark.seal.fill", GuideTheme.success)
        case "cancelled": return ("订单已取消", "xmark.circle.fill", GuideTheme.warning)
        default: return ("订单待确认", "clock.fill", GuideTheme.accent)
        }
    }
}
