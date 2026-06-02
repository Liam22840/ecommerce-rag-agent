import Foundation
import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ProductSpecLine: View {
    let product: Product

    var body: some View {
        Text(product.spec ?? "\(product.brand) · \(product.subCategory)")
            .font(.caption2)
            .foregroundStyle(GuideTheme.tertiaryInk)
            .lineLimit(1)
            .minimumScaleFactor(0.85)
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct ProductRatingSalesRow: View {
    let product: Product

    var body: some View {
        if product.rating != nil || product.sales != nil {
            HStack(spacing: 6) {
                if let rating = product.rating {
                    HStack(spacing: 1) {
                        ForEach(0..<5, id: \.self) { index in
                            Image(systemName: index < roundedStars(for: rating) ? "star.fill" : "star")
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundStyle(Color(red: 1.0, green: 0.667, blue: 0.0))
                        }

                        Text(String(format: "%.1f", rating))
                            .font(.caption2)
                            .foregroundStyle(GuideTheme.tertiaryInk)
                            .padding(.leading, 2)
                    }
                }

                if let sales = product.sales, !sales.isEmpty {
                    Text("月销\(sales)")
                        .font(.caption2)
                        .foregroundStyle(GuideTheme.tertiaryInk)
                }
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func roundedStars(for rating: Double) -> Int {
        min(5, max(0, Int(rating.rounded())))
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct ProductProsConsView: View {
    let product: Product

    var body: some View {
        if !product.pros.isEmpty || !product.cons.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(product.pros, id: \.self) { text in
                    ProductTradeoffRow(text: text, systemImage: "checkmark", tint: GuideTheme.success, background: GuideTheme.successSoft)
                }

                ForEach(product.cons, id: \.self) { text in
                    ProductTradeoffRow(text: text, systemImage: "xmark", tint: GuideTheme.warning, background: GuideTheme.warningSoft)
                }
            }
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct ProductTradeoffRow: View {
    let text: String
    let systemImage: String
    let tint: Color
    let background: Color

    var body: some View {
        HStack(alignment: .top, spacing: 5) {
            Image(systemName: systemImage)
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 14, height: 14)
                .background(background)
                .clipShape(Circle())

            Text(text)
                .font(.caption2)
                .foregroundStyle(systemImage == "checkmark" ? tint : GuideTheme.tertiaryInk)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
