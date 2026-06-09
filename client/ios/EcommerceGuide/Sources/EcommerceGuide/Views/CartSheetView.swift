import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct CartSheetView: View {
    let items: [CartItem]
    let quantityAction: (String, Int) -> Void
    let removeAction: (String) -> Void
    let checkoutAction: () -> Void

    @Environment(\.dismiss) private var dismiss

    private var total: Decimal {
        items.reduce(Decimal.zero) { partialResult, item in
            // Use the chosen SKU's unit price when the server has priced it; otherwise the base price.
            let unit = item.unitPrice.map { Decimal($0) } ?? item.product.basePrice
            return partialResult + (unit * Decimal(item.quantity))
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if items.isEmpty {
                    emptyState
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(items) { item in
                                CartItemRowView(
                                    item: item,
                                    quantityAction: quantityAction,
                                    removeAction: removeAction
                                )
                            }
                        }
                        .padding(.horizontal, 16)
                    }
                }

                if !items.isEmpty {
                    footer
                }
            }
            .background(GuideTheme.pageBackground)
            .navigationTitle("购物车")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") {
                        dismiss()
                    }
                    .foregroundStyle(GuideTheme.secondaryInk)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "bag")
                .font(.system(size: 34, weight: .semibold))
                .foregroundStyle(GuideTheme.tertiaryInk)
                .frame(width: 72, height: 72)
                .background(GuideTheme.assistantBubble)
                .clipShape(Circle())

            Text("购物车为空")
                .font(.headline.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)

            Text("从对话中加入商品后，可以在这里查看。")
                .font(.subheadline)
                .foregroundStyle(GuideTheme.secondaryInk)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 28)
        }
    }

    private var footer: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("合计")
                    .font(.caption)
                    .foregroundStyle(GuideTheme.secondaryInk)

                Text(formattedTotal)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(GuideTheme.accent)
            }

            Spacer(minLength: 12)

            Button {
                dismiss()
                checkoutAction()
            } label: {
                Text("去结算")
                    .font(.headline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 30)
                    .frame(height: 44)
                    .background(GuideTheme.accent)
                    .clipShape(Capsule())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(GuideTheme.panelBackground)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(GuideTheme.line)
                .frame(height: 1)
        }
    }

    private var formattedTotal: String {
        let value = NSDecimalNumber(decimal: total)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(total)"
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct CartItemRowView: View {
    let item: CartItem
    let quantityAction: (String, Int) -> Void
    let removeAction: (String) -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            ProductImageView(product: item.product)
                .frame(width: 52, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

            VStack(alignment: .leading, spacing: 5) {
                Text(item.product.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(GuideTheme.inkStrong)
                    .lineLimit(1)
                    .minimumScaleFactor(0.82)

                ProductSpecLine(product: item.product)

                HStack(alignment: .center) {
                    Text(item.priceLabel ?? item.product.formattedPrice)
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(GuideTheme.accent)

                    Spacer(minLength: 8)

                    quantityStepper
                }
            }

            Button {
                removeAction(item.product.id)
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(GuideTheme.tertiaryInk)
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("移除 \(item.product.title)")
        }
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(GuideTheme.line)
                .frame(height: 1)
        }
    }

    private var quantityStepper: some View {
        HStack(spacing: 0) {
            stepButton(systemImage: "minus", delta: -1, tint: GuideTheme.tertiaryInk)

            Text("\(item.quantity)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(GuideTheme.inkStrong)
                .frame(width: 28)

            stepButton(systemImage: "plus", delta: 1, tint: GuideTheme.accent)
        }
        .frame(height: 28)
        .overlay {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private func stepButton(systemImage: String, delta: Int, tint: Color) -> some View {
        Button {
            quantityAction(item.product.id, delta)
        } label: {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(delta > 0 ? "增加数量" : "减少数量")
    }
}
