import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ChatComposerView: View {
    @Binding var text: String
    let isSending: Bool
    let isListening: Bool
    let cameraAction: () -> Void
    let voiceAction: () -> Void
    let sendAction: () -> Void
    let cancelAction: () -> Void

    private var canSend: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSending && !isListening
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
                .disabled(isListening)
                .onSubmit {
                    if canSend {
                        dismissKeyboard()
                        sendAction()
                    }
                }

            Button(action: buttonAction) {
                Image(systemName: buttonIcon)
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 36, height: 36)
                    .foregroundStyle(isPrimaryButton ? .white : GuideTheme.tertiaryInk)
                    .background(isPrimaryButton ? GuideTheme.accent : GuideTheme.pageBackground)
                    .clipShape(Circle())
            }
            .disabled(isButtonDisabled)
            .accessibilityLabel(buttonAccessibilityLabel)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(GuideTheme.panelBackground)
    }

    private var buttonIcon: String {
        if isSending {
            return "stop.fill"
        }

        if isListening {
            return "stop.circle.fill"
        }

        return canSend ? "arrow.up" : "mic.fill"
    }

    private var buttonAction: () -> Void {
        if isSending {
            return cancelAction
        }

        if canSend {
            return {
                dismissKeyboard()
                sendAction()
            }
        }

        return voiceAction
    }

    private var isPrimaryButton: Bool {
        isSending || isListening || canSend
    }

    private var isButtonDisabled: Bool {
        false
    }

    private var buttonAccessibilityLabel: String {
        if isSending {
            return "停止回复"
        }

        if isListening {
            return "结束语音输入"
        }

        return canSend ? "发送消息" : "语音输入"
    }
}
