import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct CartStatusView: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(GuideTheme.success)

            Text(text)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(GuideTheme.inkStrong)
                .lineLimit(2)

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(GuideTheme.successSoft)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
    }
}

@available(iOS 17.0, macOS 13.0, *)
struct ErrorRetryView: View {
    let message: String
    let retryAction: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(GuideTheme.warning)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(GuideTheme.inkStrong)
                .lineLimit(3)

            Spacer(minLength: 8)

            Button(action: retryAction) {
                Image(systemName: "arrow.clockwise")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.plain)
            .foregroundStyle(GuideTheme.warning)
            .background(GuideTheme.warningSoft)
            .clipShape(Circle())
            .accessibilityLabel("重试")
        }
        .padding(12)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.warning.opacity(0.25))
        }
        .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)
    }
}
