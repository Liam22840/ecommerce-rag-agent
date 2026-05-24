import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct CartStatusView: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(GuideTheme.accent)

            Text(text)
                .font(.subheadline)
                .foregroundStyle(GuideTheme.ink)
                .lineLimit(2)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(GuideTheme.accentSoft)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct ErrorRetryView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(GuideTheme.warning)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(GuideTheme.ink)
                .lineLimit(3)

            Spacer(minLength: 8)

            Button(action: retryAction) {
                Label("Retry", systemImage: "arrow.clockwise")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.bordered)
            .tint(GuideTheme.warning)
        }
        .padding(12)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(GuideTheme.warning.opacity(0.35))
        }
    }
}
