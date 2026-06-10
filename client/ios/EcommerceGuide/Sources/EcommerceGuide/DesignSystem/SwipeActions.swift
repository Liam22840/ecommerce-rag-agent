import SwiftUI

@available(iOS 17.0, macOS 14.0, *)
struct SwipeAction: Identifiable {
    let systemImage: String
    let title: String
    let tint: Color
    let handler: () -> Void

    var id: String { title }
}

@available(iOS 17.0, macOS 14.0, *)
private struct GuideSwipeActionsModifier: ViewModifier {
    let itemID: String
    @Binding var openItemID: String?
    let actions: [SwipeAction]

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var offset: CGFloat = 0

    private var revealWidth: CGFloat {
        CGFloat(actions.count) * 60 + 8
    }

    func body(content: Content) -> some View {
        content
            .offset(x: offset)
            .background(alignment: .trailing) {
                actionRow
                    .opacity(offset < -10 ? 1 : 0)
            }
            .contentShape(Rectangle())
            .simultaneousGesture(closeOnTapGesture)
            .gesture(dragGesture)
            .onChange(of: openItemID) { _, newValue in
                // Another row opened (or everything closed): snap this one shut.
                if newValue != itemID && offset != 0 {
                    withAnimation(GuideMotion.reveal) {
                        offset = 0
                    }
                }
            }
            .sensoryFeedback(.impact(weight: .light), trigger: openItemID) { _, newValue in
                newValue == itemID
            }
    }

    private var actionRow: some View {
        HStack(spacing: 8) {
            ForEach(actions) { action in
                Button {
                    close()
                    action.handler()
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: action.systemImage)
                            .font(.system(size: 15, weight: .semibold))
                        Text(action.title)
                            .font(.caption2.weight(.semibold))
                    }
                    .foregroundStyle(.white)
                    .frame(width: 52, height: 52)
                    .background(action.tint, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(action.title)
            }
        }
    }

    private var dragGesture: some Gesture {
        DragGesture(minimumDistance: 24)
            .onChanged { value in
                // Axis lock: ignore drags that are mostly vertical so chat scrolling wins.
                guard abs(value.translation.width) > abs(value.translation.height) else {
                    return
                }
                let base: CGFloat = openItemID == itemID ? -revealWidth : 0
                var next = min(0, base + value.translation.width)
                if next < -revealWidth {
                    // Rubber band past the fully revealed position.
                    let overshoot = -revealWidth - next
                    next = -revealWidth - overshoot / 3
                }
                offset = next
            }
            .onEnded { _ in
                let shouldOpen = offset < -revealWidth * 0.5
                withAnimation(reduceMotion ? .easeOut(duration: 0.15) : GuideMotion.reveal) {
                    offset = shouldOpen ? -revealWidth : 0
                }
                if shouldOpen {
                    openItemID = itemID
                } else if openItemID == itemID {
                    openItemID = nil
                }
            }
    }

    private var closeOnTapGesture: some Gesture {
        TapGesture().onEnded {
            if offset != 0 {
                close()
            }
        }
    }

    private func close() {
        withAnimation(GuideMotion.reveal) {
            offset = 0
        }
        if openItemID == itemID {
            openItemID = nil
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension View {
    /// Trailing swipe-to-reveal actions for rows that live in a ScrollView (not a List).
    func guideSwipeActions(
        itemID: String,
        openItemID: Binding<String?>,
        actions: [SwipeAction]
    ) -> some View {
        modifier(GuideSwipeActionsModifier(itemID: itemID, openItemID: openItemID, actions: actions))
    }
}
