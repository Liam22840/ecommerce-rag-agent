import Foundation

enum BackendEndpoint {
    #if !targetEnvironment(simulator)
    private static let defaultStreamURL = URL(string: "http://192.168.0.176:8000/api/chat/stream")!
    #endif

    static var streamURL: URL {
        // The Simulator shares the Mac's network, so a local backend is always at loopback.
        // It ignores the shared device override so the same scheme works for both a physical
        // phone (LAN IP) and anyone running the backend alongside the Simulator.
        #if targetEnvironment(simulator)
        return URL(string: "http://127.0.0.1:8000/api/chat/stream")!
        #else
        if let value = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_BACKEND_URL"],
           let url = URL(string: value) {
            return url
        }

        if let value = UserDefaults.standard.string(forKey: "EcommerceGuideBackendURL"),
           let url = URL(string: value),
           !isStaleDeviceOverride(url) {
            return url
        }

        return defaultStreamURL
        #endif
    }

    static var baseURL: URL {
        var components = URLComponents()
        components.scheme = streamURL.scheme
        components.host = streamURL.host
        components.port = streamURL.port
        return components.url ?? URL(string: "http://192.168.0.176:8000")!
    }

    static var textToSpeechURL: URL {
        if let value = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_TTS_URL"],
           let url = URL(string: value) {
            return url
        }

        return baseURL.appending(path: "api").appending(path: "tts")
    }

    #if !targetEnvironment(simulator)
    private static func isStaleDeviceOverride(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else {
            return true
        }

        return host == "127.0.0.1"
            || host == "localhost"
            || host == "::1"
            || host == "192.168.0.184"
    }
    #endif
}
