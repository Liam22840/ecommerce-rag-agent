import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
enum GuideMotion {
    /// Soft spring for items entering the timeline.
    static let entrance = Animation.spring(response: 0.42, dampingFraction: 0.82)
    /// Tight spring for state changes (badges, toggles, status icons).
    static let snappy = Animation.spring(response: 0.3, dampingFraction: 0.75)
    /// Spring used when swipe actions snap open or closed.
    static let reveal = Animation.spring(response: 0.32, dampingFraction: 0.86)
    /// Easing for scroll-to-bottom in the chat list.
    static let scroll = Animation.easeOut(duration: 0.25)

    static func entrance(reduceMotion: Bool) -> Animation {
        reduceMotion ? .easeOut(duration: 0.12) : entrance
    }

    static func timelineInsertion(reduceMotion: Bool) -> AnyTransition {
        reduceMotion
            ? .opacity
            : .asymmetric(
                insertion: .offset(y: 12).combined(with: .opacity),
                removal: .opacity
            )
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct PressableButtonStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.97 : 1)
            .animation(GuideMotion.snappy, value: configuration.isPressed)
    }
}
