import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

@available(iOS 17.0, macOS 13.0, *)
enum GuideTheme {
    static let pageBackground = Color(red: 0.961, green: 0.961, blue: 0.953)
    static let panelBackground = Color.white
    static let ink = Color.primary
    static let inkStrong = Color(red: 0.102, green: 0.102, blue: 0.102)
    static let secondaryInk = Color(red: 0.4, green: 0.4, blue: 0.4)
    static let tertiaryInk = Color(red: 0.6, green: 0.6, blue: 0.6)
    static let line = Color(red: 0.941, green: 0.933, blue: 0.925)
    static let accent = Color(red: 0.306, green: 0.431, blue: 0.949)
    static let accentSoft = Color(red: 0.933, green: 0.945, blue: 0.996)
    static let assistantBubble = Color(red: 0.949, green: 0.945, blue: 0.937)
    static let success = Color(red: 0, green: 0.710, blue: 0.471)
    static let successSoft = Color(red: 0.902, green: 0.976, blue: 0.945)
    static let warning = Color(red: 1, green: 0.302, blue: 0.310)
    static let warningSoft = Color(red: 1, green: 0.941, blue: 0.941)
    static let favourite = Color(red: 0.937, green: 0.267, blue: 0.367)
    static let favouriteSoft = Color(red: 1, green: 0.925, blue: 0.937)

    static let cardRadius: CGFloat = 12
    static let bubbleRadius: CGFloat = 16
    static let controlRadius: CGFloat = 18

    static let cardShadow = Color.black.opacity(0.06)
    static let accentShadow = accent.opacity(0.28)

    static let currencyFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.currencyCode = "CNY"
        return formatter
    }()
}

@available(iOS 17.0, macOS 13.0, *)
extension Product {
    var formattedPrice: String {
        if let priceLabel, !priceLabel.isEmpty {
            return priceLabel
        }
        let value = NSDecimalNumber(decimal: basePrice)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(basePrice)"
    }
}
