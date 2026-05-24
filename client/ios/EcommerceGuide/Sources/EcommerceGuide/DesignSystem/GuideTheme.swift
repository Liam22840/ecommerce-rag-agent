import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

@available(iOS 17.0, macOS 14.0, *)
enum GuideTheme {
#if canImport(UIKit)
    static let pageBackground = Color(uiColor: .systemGroupedBackground)
    static let panelBackground = Color(uiColor: .secondarySystemGroupedBackground)
    static let ink = Color(uiColor: .label)
    static let secondaryInk = Color(uiColor: .secondaryLabel)
    static let line = Color(uiColor: .separator)
    static let accent = Color(red: 0.00, green: 0.46, blue: 0.42)
    static let accentSoft = Color(uiColor: .tertiarySystemFill)
#elseif canImport(AppKit)
    static let pageBackground = Color(nsColor: .windowBackgroundColor)
    static let panelBackground = Color(nsColor: .controlBackgroundColor)
    static let ink = Color(nsColor: .labelColor)
    static let secondaryInk = Color(nsColor: .secondaryLabelColor)
    static let line = Color(nsColor: .separatorColor)
    static let accent = Color(red: 0.00, green: 0.46, blue: 0.42)
    static let accentSoft = Color(nsColor: .controlColor)
#else
    static let pageBackground = Color(red: 0.96, green: 0.97, blue: 0.96)
    static let panelBackground = Color.white
    static let ink = Color.primary
    static let secondaryInk = Color.secondary
    static let line = Color.gray.opacity(0.35)
    static let accent = Color(red: 0.00, green: 0.46, blue: 0.42)
    static let accentSoft = Color.gray.opacity(0.16)
#endif
    static let warning = Color(red: 0.78, green: 0.22, blue: 0.17)

    static let currencyFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.currencyCode = "CNY"
        return formatter
    }()
}

@available(iOS 17.0, macOS 14.0, *)
extension Product {
    var formattedPrice: String {
        let value = NSDecimalNumber(decimal: basePrice)
        return GuideTheme.currencyFormatter.string(from: value) ?? "\(basePrice)"
    }
}
