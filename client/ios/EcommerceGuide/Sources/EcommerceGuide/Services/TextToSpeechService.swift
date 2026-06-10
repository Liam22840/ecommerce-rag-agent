import Foundation

public enum TextToSpeechPlaybackState: Equatable, Sendable {
    case idle
    case loading
    case speaking
}

public protocol TextToSpeechService: Sendable {
    @MainActor func speak(_ text: String)
    @MainActor func prepare(_ text: String)
    @MainActor func stopSpeaking()
    @MainActor func setPlaybackStateHandler(_ handler: (@MainActor @Sendable (TextToSpeechPlaybackState) -> Void)?)
}

#if os(iOS)
public typealias DefaultTextToSpeechService = RemoteTextToSpeechService
#else
public typealias DefaultTextToSpeechService = UnavailableTextToSpeechService
#endif

public struct UnavailableTextToSpeechService: TextToSpeechService {
    public init() {}

    public func speak(_ text: String) {}
    public func prepare(_ text: String) {}
    public func stopSpeaking() {}
    public func setPlaybackStateHandler(_ handler: (@MainActor @Sendable (TextToSpeechPlaybackState) -> Void)?) {}
}

#if os(iOS)
import AVFoundation
import CryptoKit

@available(iOS 17.0, macOS 13.0, *)
@MainActor
public final class RemoteTextToSpeechService: NSObject, TextToSpeechService, AVAudioPlayerDelegate, @unchecked Sendable {
    private let endpointURL: URL
    private let session: URLSession
    private let fallbackService: AVFoundationTextToSpeechService
    private var audioPlayer: AVAudioPlayer?
    private var requestTask: Task<Void, Never>?
    private var prepareTasks: [String: Task<Void, Never>] = [:]
    private var speechRequestID = 0
    private var audioCache: [String: Data] = [:]
    private var playbackStateHandler: (@MainActor @Sendable (TextToSpeechPlaybackState) -> Void)?

    public init(
        endpointURL: URL? = nil,
        session: URLSession = .shared,
        fallbackService: AVFoundationTextToSpeechService = AVFoundationTextToSpeechService()
    ) {
        self.endpointURL = endpointURL ?? Self.defaultEndpointURL
        self.session = session
        self.fallbackService = fallbackService
        super.init()
        self.fallbackService.completionHandler = { [weak self] in
            self?.finishCurrentSpeech()
        }
    }

    public func prepare(_ text: String) {
        let speakableText = SpeechTextPreparer.prepare(text)
        guard !speakableText.isEmpty,
              cachedAudioData(for: speakableText) == nil,
              prepareTasks[speakableText] == nil else {
            return
        }

        let endpointURL = endpointURL
        let session = session
        prepareTasks[speakableText] = Task { [weak self] in
            do {
                let audioData = try await Self.fetchAudio(
                    text: speakableText,
                    endpointURL: endpointURL,
                    session: session
                )
                try Task.checkCancellation()
                await self?.finishPrepare(audioData: audioData, for: speakableText)
            } catch {
                await self?.finishPrepare(audioData: nil, for: speakableText)
            }
        }
    }

    public func speak(_ text: String) {
        let speakableText = SpeechTextPreparer.prepare(text)
        guard !speakableText.isEmpty else {
            return
        }

        stopSpeaking()

        if let cachedAudioData = cachedAudioData(for: speakableText) {
            play(audioData: cachedAudioData, fallbackText: speakableText, requestID: nextSpeechRequestID())
            return
        }

        let endpointURL = endpointURL
        let session = session
        let requestID = nextSpeechRequestID()
        emitPlaybackState(.loading)
        requestTask = Task { [weak self] in
            do {
                let audioData = try await Self.fetchAudio(
                    text: speakableText,
                    endpointURL: endpointURL,
                    session: session
                )
                try Task.checkCancellation()
                await self?.cacheAndPlay(audioData: audioData, fallbackText: speakableText, requestID: requestID)
            } catch is CancellationError {
                await self?.finishSpeechIfCurrent(requestID)
                return
            } catch {
                guard !Task.isCancelled else {
                    await self?.finishSpeechIfCurrent(requestID)
                    return
                }
                await self?.speakWithFallback(speakableText, requestID: requestID)
            }
        }
    }

    public func stopSpeaking() {
        speechRequestID += 1
        requestTask?.cancel()
        requestTask = nil
        audioPlayer?.stop()
        audioPlayer = nil
        fallbackService.stopSpeaking()
        emitPlaybackState(.idle)
    }

    public func setPlaybackStateHandler(_ handler: (@MainActor @Sendable (TextToSpeechPlaybackState) -> Void)?) {
        playbackStateHandler = handler
        handler?(.idle)
    }

    private func finishPrepare(audioData: Data?, for text: String) {
        if let audioData {
            storeCachedAudioData(audioData, for: text)
        }
        prepareTasks[text] = nil
    }

    private func play(audioData: Data, fallbackText: String, requestID: Int) {
        guard isCurrentSpeechRequest(requestID) else {
            return
        }

        do {
            fallbackService.stopSpeaking()
            configureAudioSession()
            let player = try AVAudioPlayer(data: audioData)
            audioPlayer = player
            player.delegate = self
            player.prepareToPlay()
            if player.play() {
                finishRequest(requestID)
                emitPlaybackState(.speaking)
            } else {
                speakWithFallback(fallbackText, requestID: requestID)
            }
        } catch {
            speakWithFallback(fallbackText, requestID: requestID)
        }
    }

    private func speakWithFallback(_ text: String, requestID: Int) {
        guard isCurrentSpeechRequest(requestID) else {
            return
        }

        audioPlayer?.stop()
        audioPlayer = nil
        emitPlaybackState(.speaking)
        fallbackService.speak(text)
        finishRequest(requestID)
    }

    public nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard self?.audioPlayer === player else {
                return
            }
            self?.finishCurrentSpeech()
        }
    }

    public nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        Task { @MainActor [weak self] in
            guard self?.audioPlayer === player else {
                return
            }
            self?.finishCurrentSpeech()
        }
    }

    private func nextSpeechRequestID() -> Int {
        speechRequestID += 1
        return speechRequestID
    }

    private func isCurrentSpeechRequest(_ requestID: Int) -> Bool {
        speechRequestID == requestID
    }

    private func finishRequest(_ requestID: Int) {
        guard isCurrentSpeechRequest(requestID) else {
            return
        }

        requestTask = nil
    }

    private func finishSpeechIfCurrent(_ requestID: Int) {
        guard isCurrentSpeechRequest(requestID) else {
            return
        }

        finishCurrentSpeech()
    }

    private func finishCurrentSpeech() {
        requestTask = nil
        audioPlayer = nil
        emitPlaybackState(.idle)
    }

    private func emitPlaybackState(_ state: TextToSpeechPlaybackState) {
        playbackStateHandler?(state)
    }

    private func cacheAndPlay(audioData: Data, fallbackText: String, requestID: Int) {
        guard isCurrentSpeechRequest(requestID) else {
            return
        }

        storeCachedAudioData(audioData, for: fallbackText)
        play(audioData: audioData, fallbackText: fallbackText, requestID: requestID)
    }

    private func cachedAudioData(for text: String) -> Data? {
        if let data = audioCache[text] {
            return data
        }

        if let cacheURL = Self.diskCacheURL(for: text),
           let data = try? Data(contentsOf: cacheURL),
           !data.isEmpty {
            audioCache[text] = data
            return data
        }

        if let data = Self.bundledAudioData(for: text) {
            audioCache[text] = data
            return data
        }

        return nil
    }

    private func storeCachedAudioData(_ data: Data, for text: String) {
        audioCache[text] = data

        guard let cacheURL = Self.diskCacheURL(for: text) else {
            return
        }

        do {
            try FileManager.default.createDirectory(
                at: cacheURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: cacheURL, options: .atomic)
        } catch {
            return
        }
    }

    private nonisolated static func fetchAudio(
        text: String,
        endpointURL: URL,
        session: URLSession
    ) async throws -> Data {
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("audio/wav", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 45
        request.httpBody = try JSONEncoder().encode(TextToSpeechPayload(text: text))

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode),
              !data.isEmpty else {
            throw ChatServiceError.invalidResponse
        }

        return data
    }

    private func configureAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? audioSession.setActive(true)
    }

    public static var defaultEndpointURL: URL {
        if let value = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_TTS_URL"],
           let url = URL(string: value) {
            return url
        }

        let configured = ProcessInfo.processInfo.environment["ECOMMERCE_GUIDE_BACKEND_URL"]
            ?? UserDefaults.standard.string(forKey: "EcommerceGuideBackendURL")
                .flatMap { $0.contains("192.168.0.184") ? nil : $0 }
        let endpoint = configured.flatMap { URL(string: $0) }
            ?? URL(string: "http://192.168.0.176:8000/api/chat/stream")!

        var components = URLComponents()
        components.scheme = endpoint.scheme
        components.host = endpoint.host
        components.port = endpoint.port
        components.path = "/api/tts"
        return components.url ?? URL(string: "http://192.168.0.176:8000/api/tts")!
    }

    private nonisolated static func diskCacheURL(for text: String) -> URL? {
        guard let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return nil
        }

        return directory
            .appendingPathComponent("EcommerceGuideTTS", isDirectory: true)
            .appendingPathComponent("\(cacheKey(for: text)).wav")
    }

    private nonisolated static func cacheKey(for text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private nonisolated static func bundledAudioData(for text: String) -> Data? {
        guard let resourceName = bundledAudioResourceNames[text] else {
            return nil
        }

        let resourceURL = Bundle.module.url(
            forResource: resourceName,
            withExtension: "wav",
            subdirectory: "Audio"
        ) ?? Bundle.module.url(forResource: resourceName, withExtension: "wav")

        guard let resourceURL,
              let data = try? Data(contentsOf: resourceURL),
              !data.isEmpty else {
            return nil
        }

        return data
    }

    private nonisolated static let bundledAudioResourceNames: [String: String] = [
        "你好，我是你的 A I 购物助手。告诉我你想买什么，我会帮你对比商品、解释取舍，并整理好购物车。": "welcome-message"
    ]
}

public final class AVFoundationTextToSpeechService: NSObject, TextToSpeechService, AVSpeechSynthesizerDelegate, @unchecked Sendable {
    private let synthesizer = AVSpeechSynthesizer()
    private let localeIdentifier: String
    var completionHandler: (@MainActor @Sendable () -> Void)?

    public init(localeIdentifier: String = "zh-CN") {
        self.localeIdentifier = localeIdentifier
        super.init()
        synthesizer.delegate = self
    }

    @MainActor
    public func speak(_ text: String) {
        let speakableText = SpeechTextPreparer.prepare(text)
        guard !speakableText.isEmpty else {
            return
        }

        stopSpeaking()
        configureAudioSession()

        let utterance = AVSpeechUtterance(string: speakableText)
        utterance.voice = preferredVoice()
        utterance.rate = 0.46
        utterance.pitchMultiplier = 1.04
        utterance.volume = 1
        utterance.preUtteranceDelay = 0.08
        utterance.postUtteranceDelay = 0.12
        synthesizer.speak(utterance)
    }

    @MainActor
    public func prepare(_ text: String) {}

    @MainActor
    public func stopSpeaking() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
    }

    public func setPlaybackStateHandler(_ handler: (@MainActor @Sendable (TextToSpeechPlaybackState) -> Void)?) {}

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.completionHandler?()
        }
    }

    public nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.completionHandler?()
        }
    }

    private func preferredVoice() -> AVSpeechSynthesisVoice? {
        let availableVoices = AVSpeechSynthesisVoice.speechVoices()
        let localeVoices = availableVoices.filter { voice in
            voice.language == localeIdentifier || voice.language.hasPrefix("zh")
        }

        return localeVoices.max { lhs, rhs in
            Self.voiceScore(lhs, localeIdentifier: localeIdentifier) < Self.voiceScore(rhs, localeIdentifier: localeIdentifier)
        } ?? AVSpeechSynthesisVoice(language: localeIdentifier)
    }

    private static func voiceScore(_ voice: AVSpeechSynthesisVoice, localeIdentifier: String) -> Int {
        var score = 0
        if voice.language == localeIdentifier {
            score += 20
        } else if voice.language.hasPrefix("zh") {
            score += 10
        }

        switch voice.quality {
        case .premium:
            score += 3
        case .enhanced:
            score += 2
        case .default:
            score += 1
        @unknown default:
            break
        }

        return score
    }

    private func configureAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? audioSession.setActive(true)
    }
}

private struct TextToSpeechPayload: Encodable {
    let text: String
}

private enum SpeechTextPreparer {
    static func prepare(_ text: String) -> String {
        text
            .replacingOccurrences(of: "SKU", with: "规格")
            .replacingOccurrences(of: "AI", with: "A I")
            .replacingOccurrences(of: "；", with: "。")
            .replacingOccurrences(of: ";", with: "。")
            .replacingOccurrences(of: "：", with: "，")
            .replacingOccurrences(of: ":", with: "，")
            .replacingOccurrences(of: "\n", with: "。")
            .replacingOccurrences(of: #"[*_`#>\-]+"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
#endif
