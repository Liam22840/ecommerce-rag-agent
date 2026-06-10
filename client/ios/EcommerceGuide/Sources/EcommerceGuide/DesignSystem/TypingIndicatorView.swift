import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct TypingIndicatorView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animating = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(GuideTheme.secondaryInk)
                    .frame(width: 6, height: 6)
                    .offset(y: animating ? -3 : 2)
                    .animation(
                        reduceMotion
                            ? nil
                            : .easeInOut(duration: 0.45)
                                .repeatForever(autoreverses: true)
                                .delay(Double(index) * 0.15),
                        value: animating
                    )
            }
        }
        .padding(.vertical, 4)
        .onAppear {
            animating = true
        }
        .accessibilityLabel("正在思考")
    }
}
