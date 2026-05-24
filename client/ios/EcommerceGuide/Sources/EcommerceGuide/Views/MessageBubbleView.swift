import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct MessageBubbleView: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 42)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text(message.text.isEmpty ? "Thinking..." : message.text)
                    .font(.body)
                    .foregroundStyle(message.role == .user ? .white : GuideTheme.ink)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)

                if message.isStreaming {
                    ProgressView()
                        .controlSize(.small)
                        .tint(message.role == .user ? .white : GuideTheme.accent)
                }
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 10)
            .background(message.role == .user ? GuideTheme.accent : GuideTheme.panelBackground)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay {
                if message.role == .assistant {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(GuideTheme.line)
                }
            }
            .frame(maxWidth: 610, alignment: message.role == .user ? .trailing : .leading)
            .dynamicTypeSize(...DynamicTypeSize.accessibility3)

            if message.role == .assistant {
                Spacer(minLength: 42)
            }
        }
    }
}
