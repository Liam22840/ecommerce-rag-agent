import SwiftUI

/// Collects the rects needed for the add-to-cart flight: each visible product card image
/// (keyed by product ID) and the header cart pill.
@available(iOS 17.0, macOS 14.0, *)
struct GuideAnchorKey: PreferenceKey {
    struct Value {
        var productImages: [String: Anchor<CGRect>] = [:]
        var cartPill: Anchor<CGRect>?
    }

    static let defaultValue = Value()

    static func reduce(value: inout Value, nextValue: () -> Value) {
        let next = nextValue()
        value.productImages.merge(next.productImages) { _, new in new }
        if let pill = next.cartPill {
            value.cartPill = pill
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
struct CartFlight: Identifiable, Equatable {
    let id: UUID
    let product: Product
}

@available(iOS 17.0, macOS 14.0, *)
struct CartFlightView: View {
    let flight: CartFlight
    // `from`/`to` must be in the coordinate space of this view's direct parent: resolve the
    // anchors with the GeometryProxy of the overlay this view is placed in (it uses .position).
    let from: CGPoint
    let to: CGPoint
    let onFinished: () -> Void

    @State private var progress: CGFloat = 0

    // Teardown waits slightly past the animation so the thumbnail lands before removal.
    private static let flightDuration: TimeInterval = 0.55

    var body: some View {
        ProductImageView(product: flight.product)
            .frame(width: 44, height: 44)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .modifier(FlightPathModifier(progress: progress, from: from, to: to))
            .onAppear {
                withAnimation(.easeIn(duration: Self.flightDuration)) {
                    progress = 1
                }
            }
            .task {
                try? await Task.sleep(nanoseconds: UInt64((Self.flightDuration + 0.1) * 1_000_000_000))
                onFinished()
            }
    }
}

/// Moves content along a quadratic Bezier from `from` to `to` while shrinking and fading.
@available(iOS 17.0, macOS 14.0, *)
private struct FlightPathModifier: ViewModifier, Animatable {
    var progress: CGFloat
    let from: CGPoint
    let to: CGPoint

    var animatableData: CGFloat {
        get { progress }
        set { progress = newValue }
    }

    func body(content: Content) -> some View {
        let control = CGPoint(x: (from.x + to.x) / 2, y: min(from.y, to.y) - 80)
        let t = progress
        let x = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * control.x + t * t * to.x
        let y = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * control.y + t * t * to.y

        content
            .scaleEffect(1 - 0.65 * t)
            .opacity(Double(1 - 0.4 * t))
            .position(x: x, y: y)
    }
}
