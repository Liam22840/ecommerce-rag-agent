import SwiftUI
#if canImport(UIKit)
import UIKit
private typealias GuidePlatformColor = UIColor
#elseif canImport(AppKit)
import AppKit
private typealias GuidePlatformColor = NSColor
#endif

@available(iOS 17.0, macOS 13.0, *)
enum GuideTheme {
    static let pageBackground = adaptive(
        light: color(0.961, 0.961, 0.953),
        dark: color(0.071, 0.073, 0.078)
    )
    static let panelBackground = adaptive(
        light: color(1, 1, 1),
        dark: color(0.118, 0.122, 0.133)
    )
    static let ink = Color.primary
    static let inkStrong = adaptive(
        light: color(0.102, 0.102, 0.102),
        dark: color(0.941, 0.945, 0.953)
    )
    static let secondaryInk = adaptive(
        light: color(0.4, 0.4, 0.4),
        dark: color(0.718, 0.733, 0.765)
    )
    static let tertiaryInk = adaptive(
        light: color(0.6, 0.6, 0.6),
        dark: color(0.529, 0.545, 0.584)
    )
    static let line = adaptive(
        light: color(0.941, 0.933, 0.925),
        dark: color(0.235, 0.243, 0.263)
    )
    static let accent = adaptive(
        light: color(0.306, 0.431, 0.949),
        dark: color(0.494, 0.604, 1)
    )
    static let accentSoft = adaptive(
        light: color(0.933, 0.945, 0.996),
        dark: color(0.098, 0.133, 0.286)
    )
    static let assistantBubble = adaptive(
        light: color(0.949, 0.945, 0.937),
        dark: color(0.157, 0.161, 0.173)
    )
    static let success = adaptive(
        light: color(0, 0.710, 0.471),
        dark: color(0.282, 0.831, 0.604)
    )
    static let successSoft = adaptive(
        light: color(0.902, 0.976, 0.945),
        dark: color(0.063, 0.220, 0.157)
    )
    static let warning = adaptive(
        light: color(1, 0.302, 0.310),
        dark: color(1, 0.424, 0.424)
    )
    static let warningSoft = adaptive(
        light: color(1, 0.941, 0.941),
        dark: color(0.298, 0.118, 0.129)
    )
    static let favourite = adaptive(
        light: color(0.937, 0.267, 0.367),
        dark: color(1, 0.420, 0.510)
    )
    static let favouriteSoft = adaptive(
        light: color(1, 0.925, 0.937),
        dark: color(0.286, 0.098, 0.145)
    )

    static let cardRadius: CGFloat = 12
    static let bubbleRadius: CGFloat = 16
    static let controlRadius: CGFloat = 18

    static let cardShadow = adaptive(
        light: color(0, 0, 0, alpha: 0.06),
        dark: color(0, 0, 0, alpha: 0.28)
    )
    static let accentShadow = accent.opacity(0.28)

    static let currencyFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        formatter.locale = Locale(identifier: "zh_CN")
        formatter.currencyCode = "CNY"
        return formatter
    }()

    private static func color(
        _ red: CGFloat,
        _ green: CGFloat,
        _ blue: CGFloat,
        alpha: CGFloat = 1
    ) -> GuidePlatformColor {
        #if canImport(UIKit)
        UIColor(red: red, green: green, blue: blue, alpha: alpha)
        #elseif canImport(AppKit)
        NSColor(srgbRed: red, green: green, blue: blue, alpha: alpha)
        #endif
    }

    private static func adaptive(light: GuidePlatformColor, dark: GuidePlatformColor) -> Color {
        #if canImport(UIKit)
        Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark ? dark : light
        })
        #elseif canImport(AppKit)
        Color(nsColor: NSColor(name: nil) { appearance in
            let match = appearance.bestMatch(from: [.darkAqua, .aqua, .vibrantDark, .vibrantLight])
            return match == .darkAqua || match == .vibrantDark ? dark : light
        })
        #endif
    }
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
