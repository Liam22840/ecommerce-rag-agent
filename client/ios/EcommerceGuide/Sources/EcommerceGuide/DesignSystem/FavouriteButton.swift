import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct FavouriteButton: View {
    let isFavourite: Bool
    var compact = false
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var burstTrigger = 0

    var body: some View {
        Button {
            if !isFavourite {
                burstTrigger += 1
            }
            action()
        } label: {
            Image(systemName: isFavourite ? "heart.fill" : "heart")
                .font(.system(size: compact ? 12 : 16, weight: .semibold))
                .foregroundStyle(isFavourite ? GuideTheme.favourite : GuideTheme.secondaryInk)
                .frame(width: compact ? 26 : 34, height: compact ? 26 : 34)
                .background(.thinMaterial, in: Circle())
                .contentTransition(.symbolEffect(.replace))
                .symbolEffect(.bounce, value: burstTrigger)
        }
        .buttonStyle(.plain)
        .overlay {
            if !reduceMotion {
                HeartBurstView(trigger: burstTrigger)
            }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: burstTrigger)
        .animation(GuideMotion.snappy, value: isFavourite)
        .accessibilityLabel(isFavourite ? "取消收藏" : "收藏")
    }
}

@available(iOS 17.0, macOS 14.0, *)
private struct HeartBurstView: View {
    let trigger: Int

    private struct BurstState {
        var distance: CGFloat = 0
        var opacity: Double = 0
        var scale: CGFloat = 0.4
    }

    var body: some View {
        ZStack {
            ForEach(0..<6, id: \.self) { index in
                let angle = Double(index) / 6 * 2 * .pi
                Image(systemName: "heart.fill")
                    .font(.system(size: 8))
                    .foregroundStyle(GuideTheme.favourite)
                    .keyframeAnimator(initialValue: BurstState(), trigger: trigger) { view, state in
                        view
                            .offset(x: cos(angle) * state.distance, y: sin(angle) * state.distance)
                            .opacity(state.opacity)
                            .scaleEffect(state.scale)
                    } keyframes: { _ in
                        KeyframeTrack(\.distance) {
                            CubicKeyframe(26, duration: 0.45)
                        }
                        KeyframeTrack(\.opacity) {
                            LinearKeyframe(1, duration: 0.05)
                            LinearKeyframe(0, duration: 0.4)
                        }
                        KeyframeTrack(\.scale) {
                            CubicKeyframe(1, duration: 0.1)
                            CubicKeyframe(0.4, duration: 0.35)
                        }
                    }
            }
        }
        .allowsHitTesting(false)
    }
}
