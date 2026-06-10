import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct MessageBubbleView: View {
    let message: ChatMessage
    let speechPlaybackState: TextToSpeechPlaybackState?
    let speakAction: (ChatMessage) -> Void
    let stopSpeechAction: () -> Void

    init(
        message: ChatMessage,
        speechPlaybackState: TextToSpeechPlaybackState? = nil,
        speakAction: @escaping (ChatMessage) -> Void = { _ in },
        stopSpeechAction: @escaping () -> Void = {}
    ) {
        self.message = message
        self.speechPlaybackState = speechPlaybackState
        self.speakAction = speakAction
        self.stopSpeechAction = stopSpeechAction
    }

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

                if message.text.isEmpty && message.isStreaming && message.imageData == nil {
                    TypingIndicatorView()
                } else if !(message.imageData != nil && message.text.isEmpty) {
                    Text(message.text)
                        .font(.subheadline)
                        .lineSpacing(2)
                        .foregroundStyle(message.role == .user ? .white : GuideTheme.inkStrong)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }

                if message.role == .assistant, !message.isStreaming, !message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    speechControls
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

    @ViewBuilder
    private var speechControls: some View {
        Button {
            if speechPlaybackState == nil {
                speakAction(message)
            } else {
                stopSpeechAction()
            }
        } label: {
            ZStack {
                Circle()
                    .fill(speechPlaybackState == nil ? GuideTheme.accentSoft : GuideTheme.accent)
                    .frame(width: 28, height: 28)

                switch speechPlaybackState {
                case .loading:
                    ProgressView()
                        .controlSize(.small)
                        .tint(.white)
                case .speaking:
                    Image(systemName: "stop.fill")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.white)
                case .idle, .none:
                    Image(systemName: "speaker.wave.2.fill")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(GuideTheme.accent)
                }
            }
            .frame(width: 28, height: 28)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
    }

    private var accessibilityLabel: String {
        switch speechPlaybackState {
        case .loading:
            return "停止生成语音"
        case .speaking:
            return "停止朗读"
        case .idle, .none:
            return "朗读 AI 回复"
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

func platformImage(data: Data) -> Image? {
    UIImage(data: data).map { Image(uiImage: $0) }
}
#elseif canImport(AppKit)
import AppKit

func platformImage(data: Data) -> Image? {
    NSImage(data: data).map { Image(nsImage: $0) }
}
#else
func platformImage(data: Data) -> Image? { nil }
#endif
