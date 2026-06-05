import SwiftUI

@available(iOS 17.0, macOS 13.0, *)
public struct EcommerceGuideApp: App {
    private let service: any ChatService

    public init() {
        self.service = MockChatService()
    }

    public init(service: any ChatService) {
        self.service = service
    }

    public var body: some Scene {
        WindowGroup {
            ShoppingConciergeRootView(service: service)
        }
    }
}
