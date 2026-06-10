import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct ProductCardsSkeleton: View {
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            AssistantAvatarView()

            VStack(spacing: 6) {
                skeletonCard
                skeletonCard
            }
            .frame(maxWidth: 610, alignment: .leading)

            Spacer(minLength: 42)
        }
        .accessibilityLabel("正在加载商品")
    }

    private var skeletonCard: some View {
        HStack(alignment: .top, spacing: 10) {
            SkeletonBlock(cornerRadius: 10)
                .frame(width: 72, height: 72)

            VStack(alignment: .leading, spacing: 8) {
                SkeletonBlock()
                    .frame(width: 150, height: 13)
                SkeletonBlock()
                    .frame(width: 100, height: 10)
                SkeletonBlock()
                    .frame(width: 180, height: 10)

                HStack {
                    SkeletonBlock()
                        .frame(width: 64, height: 16)
                    Spacer()
                    SkeletonBlock(cornerRadius: 12)
                        .frame(width: 58, height: 24)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(GuideTheme.panelBackground)
        .shimmer()
        .clipShape(RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: GuideTheme.cardRadius, style: .continuous)
                .stroke(Color.black.opacity(0.04))
        }
        .shadow(color: GuideTheme.cardShadow, radius: 3, y: 1)
    }
}
