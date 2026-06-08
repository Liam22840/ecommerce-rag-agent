import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct MessageBubbleView: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if message.role == .user {
                Spacer(minLength: 42)
            } else {
                AssistantAvatarView()
            }

            VStack(alignment: .leading, spacing: 6) {
                if let imageData = message.imageData, let image = platformImage(data: imageData) {
                    image
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: 180, maxHeight: 180)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }

                if !(message.imageData != nil && message.text.isEmpty) {
                    Text(message.text.isEmpty ? "正在思考..." : message.text)
                        .font(.subheadline)
                        .lineSpacing(2)
                        .foregroundStyle(message.role == .user ? .white : GuideTheme.inkStrong)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }

                if message.isStreaming {
                    ProgressView()
                        .controlSize(.small)
                        .tint(message.role == .user ? .white : GuideTheme.accent)
                }
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 10)
            .background(message.role == .user ? GuideTheme.accent : GuideTheme.assistantBubble)
            .clipShape(RoundedRectangle(cornerRadius: GuideTheme.bubbleRadius, style: .continuous))
            .frame(maxWidth: 610, alignment: message.role == .user ? .trailing : .leading)
            .dynamicTypeSize(...DynamicTypeSize.accessibility3)
            .shadow(color: message.role == .user ? GuideTheme.accentShadow.opacity(0.2) : .clear, radius: 8, y: 3)

            if message.role == .assistant {
                Spacer(minLength: 42)
            }
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct AssistantAvatarView: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [GuideTheme.accent, Color(red: 0.482, green: 0.576, blue: 0.969)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Image(systemName: "sparkle")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(.white)
        }
        .frame(width: 32, height: 32)
        .clipShape(Circle())
        .shadow(color: GuideTheme.accentShadow, radius: 6, y: 2)
        .accessibilityHidden(true)
    }
}

#if canImport(UIKit)
import UIKit

private func platformImage(data: Data) -> Image? {
    UIImage(data: data).map { Image(uiImage: $0) }
}
#elseif canImport(AppKit)
import AppKit

private func platformImage(data: Data) -> Image? {
    NSImage(data: data).map { Image(nsImage: $0) }
}
#else
private func platformImage(data: Data) -> Image? { nil }
#endif
