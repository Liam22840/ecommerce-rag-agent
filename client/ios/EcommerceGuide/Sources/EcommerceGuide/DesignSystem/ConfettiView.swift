import SwiftUI

/// One-shot confetti burst rendered with Canvas + TimelineView. Decorative only:
/// renders nothing when reduce motion is on.
@available(iOS 17.0, macOS 14.0, *)
struct ConfettiView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var start: Date?
    @State private var isDone = false

    private struct Particle {
        let xRatio: CGFloat
        let delay: Double
        let fallSpeed: CGFloat
        let drift: CGFloat
        let spin: Double
        let size: CGSize
        let color: Color
    }

    private static let palette: [Color] = [
        GuideTheme.accent,
        GuideTheme.success,
        GuideTheme.favourite,
        Color(red: 1.0, green: 0.78, blue: 0.25)
    ]

    private let duration: Double = 2.4

    private static let particles: [Particle] = (0..<70).map { _ in
        Particle(
            xRatio: .random(in: 0.02...0.98),
            delay: .random(in: 0...0.35),
            fallSpeed: .random(in: 240...420),
            drift: .random(in: -60...60),
            spin: .random(in: -6...6),
            size: CGSize(width: .random(in: 5...9), height: .random(in: 8...13)),
            color: palette.randomElement() ?? GuideTheme.accent
        )
    }

    var body: some View {
        if !reduceMotion {
            // Paused once the burst ends so the display-link stops ticking on a no-op canvas.
            TimelineView(.animation(minimumInterval: nil, paused: isDone)) { context in
                Canvas { canvas, size in
                    guard let start else {
                        return
                    }
                    let elapsed = context.date.timeIntervalSince(start)
                    guard elapsed < duration else {
                        return
                    }

                    for particle in Self.particles {
                        let t = elapsed - particle.delay
                        guard t > 0 else {
                            continue
                        }
                        let x = particle.xRatio * size.width + particle.drift * t
                        let y = -20 + particle.fallSpeed * t
                        guard y < size.height + 20 else {
                            continue
                        }

                        var ctx = canvas
                        ctx.translateBy(x: x, y: y)
                        ctx.rotate(by: .radians(particle.spin * t))
                        ctx.opacity = min(1, max(0, (duration - elapsed) / 0.5))
                        ctx.fill(
                            Path(
                                roundedRect: CGRect(
                                    origin: CGPoint(x: -particle.size.width / 2, y: -particle.size.height / 2),
                                    size: particle.size
                                ),
                                cornerRadius: 1.5
                            ),
                            with: .color(particle.color)
                        )
                    }
                }
            }
            .allowsHitTesting(false)
            .onAppear {
                start = Date()
            }
            .task {
                try? await Task.sleep(nanoseconds: UInt64((duration + 0.2) * 1_000_000_000))
                isDone = true
            }
        }
    }
}
