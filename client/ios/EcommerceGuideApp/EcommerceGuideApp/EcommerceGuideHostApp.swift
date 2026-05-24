import EcommerceGuide
import SwiftUI

@main
struct EcommerceGuideHostApp: App {
    var body: some Scene {
        WindowGroup {
            ChatScreen(viewModel: ChatViewModel(service: configuredService))
        }
    }

    private var configuredService: any ChatService {
        let value = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_SERVICE"]
            ?? UserDefaults.standard.string(forKey: "EcommerceGuideService")

        if value?.lowercased() == "sse" {
            return SSEChatService()
        }

        return MockChatService()
    }
}
