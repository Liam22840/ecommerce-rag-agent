import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct PlanStatusView: View {
    let steps: [PlanStep]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(steps, id: \.stepID) { step in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: iconName(for: step.status))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(tint(for: step.status))
                        .frame(width: 18, height: 18)

                    VStack(alignment: .leading, spacing: 3) {
                        Text(step.title)
                            .font(.footnote.weight(.medium))
                            .foregroundStyle(titleTint(for: step.status))
                            .lineLimit(2)

                        if let summary = step.summary, !summary.isEmpty {
                            Text(summary)
                                .font(.caption2)
                                .foregroundStyle(GuideTheme.secondaryInk)
                                .lineLimit(2)
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: 320, alignment: .leading)
        .background(GuideTheme.assistantBubble)
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(GuideTheme.line)
        }
    }

    private func iconName(for status: String) -> String {
        switch status {
        case "done":
            return "checkmark.circle.fill"
        case "running":
            return "circle.dotted"
        case "failed":
            return "exclamationmark.circle.fill"
        case "skipped":
            return "minus.circle.fill"
        default:
            return "circle"
        }
    }

    private func tint(for status: String) -> Color {
        switch status {
        case "done":
            return GuideTheme.success
        case "failed":
            return GuideTheme.warning
        case "running":
            return GuideTheme.accent
        default:
            return GuideTheme.tertiaryInk
        }
    }

    private func titleTint(for status: String) -> Color {
        switch status {
        case "pending", "skipped":
            return GuideTheme.tertiaryInk
        default:
            return GuideTheme.secondaryInk
        }
    }
}
