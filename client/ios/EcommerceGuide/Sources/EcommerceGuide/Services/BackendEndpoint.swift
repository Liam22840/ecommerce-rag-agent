import Foundation

enum BackendEndpoint {
    private static let defaultStreamURL = URL(string: "http://192.168.0.176:8000/api/chat/stream")!

    static var streamURL: URL {
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

    private static func isStaleDeviceOverride(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else {
            return true
        }

        return host == "127.0.0.1"
            || host == "localhost"
            || host == "::1"
            || host == "192.168.0.184"
    }
}
