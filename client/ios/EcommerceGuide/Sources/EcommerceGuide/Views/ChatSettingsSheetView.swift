import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
struct ChatSettingsSheetView: View {
    @Binding var isAutoReadingEnabled: Bool
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                SettingsToggleRow(
                    systemImage: isAutoReadingEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill",
                    title: "自动朗读",
                    isOn: $isAutoReadingEnabled
                )

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .background(GuideTheme.pageBackground)
            .navigationTitle("设置")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("关闭") {
                        dismiss()
                    }
                    .foregroundStyle(GuideTheme.secondaryInk)
                }
            }
        }
    }
}

@available(iOS 17.0, macOS 13.0, *)
private struct SettingsToggleRow: View {
    let systemImage: String
    let title: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(isOn: $isOn) {
            HStack(spacing: 12) {
                Image(systemName: systemImage)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(isOn ? .white : GuideTheme.accent)
                    .frame(width: 34, height: 34)
                    .background(isOn ? GuideTheme.accent : GuideTheme.accentSoft)
                    .clipShape(Circle())
                    .contentTransition(.symbolEffect(.replace))

                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(GuideTheme.inkStrong)
            }
        }
        .toggleStyle(.switch)
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(GuideTheme.panelBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .accessibilityValue(isOn ? "已开启" : "已关闭")
    }
}
