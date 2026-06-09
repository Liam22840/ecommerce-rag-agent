import Foundation

public struct SpeechRecognitionUpdate: Equatable, Sendable {
    public let transcript: String
    public let isFinal: Bool

    public init(transcript: String, isFinal: Bool = false) {
        self.transcript = transcript
        self.isFinal = isFinal
    }
}

public enum SpeechRecognitionError: LocalizedError, Equatable, Sendable {
    case unavailable
    case authorizationDenied
    case authorizationRestricted
    case microphoneDenied
    case recognizerUnavailable
    case audioSessionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unavailable:
            return "当前设备不支持语音输入。"
        case .authorizationDenied:
            return "请在系统设置中允许语音识别权限。"
        case .authorizationRestricted:
            return "当前设备无法使用语音识别。"
        case .microphoneDenied:
            return "请在系统设置中允许麦克风权限。"
        case .recognizerUnavailable:
            return "普通话语音识别暂时不可用，请稍后再试。"
        case .audioSessionFailed(let message):
            return "语音输入启动失败：\(message)"
        }
    }
}

public protocol SpeechRecognitionService: Sendable {
    func startMandarinRecognition() -> AsyncThrowingStream<SpeechRecognitionUpdate, Error>
    func stopRecognition()
}

#if os(iOS)
public typealias DefaultSpeechRecognitionService = MandarinSpeechRecognitionService
#else
public typealias DefaultSpeechRecognitionService = UnavailableSpeechRecognitionService
#endif

public struct UnavailableSpeechRecognitionService: SpeechRecognitionService {
    public init() {}

    public func startMandarinRecognition() -> AsyncThrowingStream<SpeechRecognitionUpdate, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: SpeechRecognitionError.unavailable)
        }
    }

    public func stopRecognition() {}
}
