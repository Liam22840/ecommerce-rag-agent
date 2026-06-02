import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ChatComposerView: View {
    @Binding var text: String
    let isSending: Bool
    let cameraAction: () -> Void
    let sendAction: () -> Void
    let cancelAction: () -> Void

    private var canSend: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSending
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Button(action: cameraAction) {
                Image(systemName: "camera.fill")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(GuideTheme.tertiaryInk)
                    .frame(width: 36, height: 36)
                    .background(GuideTheme.pageBackground)
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("相机")

            TextField("说出你想买什么...", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .font(.subheadline)
                .foregroundStyle(GuideTheme.inkStrong)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .frame(minHeight: 36)
                .background(GuideTheme.pageBackground)
                .clipShape(RoundedRectangle(cornerRadius: GuideTheme.controlRadius, style: .continuous))
                .submitLabel(.send)
                .onSubmit {
                    if canSend {
                        sendAction()
                    }
                }

            Button(action: isSending ? cancelAction : sendAction) {
                Image(systemName: buttonIcon)
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 36, height: 36)
                    .foregroundStyle(canSend || isSending ? .white : GuideTheme.tertiaryInk)
                    .background(canSend || isSending ? GuideTheme.accent : GuideTheme.pageBackground)
                    .clipShape(Circle())
            }
            .disabled(!isSending && !canSend)
            .accessibilityLabel(isSending ? "停止回复" : "发送消息")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(GuideTheme.panelBackground)
    }

    private var buttonIcon: String {
        if isSending {
            return "stop.fill"
        }

        return canSend ? "arrow.up" : "mic.fill"
    }
}
