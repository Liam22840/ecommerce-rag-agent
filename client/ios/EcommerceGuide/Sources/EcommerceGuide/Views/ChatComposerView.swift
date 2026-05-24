import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct ChatComposerView: View {
    @Binding var text: String
    let isSending: Bool
    let sendAction: () -> Void
    let cancelAction: () -> Void

    private var canSend: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isSending
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Ask for products, comparisons, or cart help", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...4)
                .font(.body)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .frame(minHeight: 44)
                .background(GuideTheme.pageBackground)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(GuideTheme.line)
                }
                .submitLabel(.send)
                .onSubmit {
                    if canSend {
                        sendAction()
                    }
                }

            Button(action: isSending ? cancelAction : sendAction) {
                Image(systemName: isSending ? "stop.fill" : "arrow.up")
                    .font(.system(size: 15, weight: .bold))
                    .frame(width: 44, height: 44)
                    .foregroundStyle(.white)
                    .background(isSending || canSend ? GuideTheme.accent : GuideTheme.line)
                    .clipShape(Circle())
            }
            .disabled(!isSending && !canSend)
            .accessibilityLabel(isSending ? "Stop response" : "Send message")
        }
        .padding(12)
        .background(.regularMaterial)
    }
}
