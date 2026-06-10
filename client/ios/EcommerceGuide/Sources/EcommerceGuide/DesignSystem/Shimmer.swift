import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
private struct ShimmerModifier: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: CGFloat = -1

    func body(content: Content) -> some View {
        if reduceMotion {
            content.opacity(0.6)
        } else {
            content
                .overlay {
                    GeometryReader { proxy in
                        LinearGradient(
                            colors: [.clear, .white.opacity(0.55), .clear],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                        .frame(width: proxy.size.width * 0.6)
                        .offset(x: phase * proxy.size.width)
                    }
                }
                .clipped()
                .onAppear {
                    phase = -1
                    withAnimation(.linear(duration: 1.1).repeatForever(autoreverses: false)) {
                        phase = 1
                    }
                }
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension View {
    /// A moving highlight sweep for skeleton placeholders.
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct SkeletonBlock: View {
    var cornerRadius: CGFloat = 6

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .fill(GuideTheme.assistantBubble)
    }
}
