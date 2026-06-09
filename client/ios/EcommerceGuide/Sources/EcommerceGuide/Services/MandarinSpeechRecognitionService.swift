#if os(iOS)
import AVFoundation
import Foundation
import Speech

public final class MandarinSpeechRecognitionService: NSObject, SpeechRecognitionService, @unchecked Sendable {
    private let locale: Locale
    private var speechRecognizer: SFSpeechRecognizer?
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    public init(locale: Locale = Locale(identifier: "zh-CN")) {
        self.locale = locale
        super.init()
    }

    public func startMandarinRecognition() -> AsyncThrowingStream<SpeechRecognitionUpdate, Error> {
        AsyncThrowingStream { continuation in
            let startupTask = Task {
                do {
                    try await requestSpeechAuthorization()
                    try await requestMicrophoneAuthorization()
                    try startAudioRecognition(continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { [weak self] _ in
                startupTask.cancel()
                self?.stopRecognition()
            }
        }
    }

    public func stopRecognition() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()

        audioEngine = nil
        recognitionRequest = nil
        recognitionTask = nil
        speechRecognizer = nil

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func requestSpeechAuthorization() async throws {
        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        switch status {
        case .authorized:
            return
        case .denied:
            throw SpeechRecognitionError.authorizationDenied
        case .restricted:
            throw SpeechRecognitionError.authorizationRestricted
        case .notDetermined:
            throw SpeechRecognitionError.authorizationDenied
        @unknown default:
            throw SpeechRecognitionError.unavailable
        }
    }

    private func requestMicrophoneAuthorization() async throws {
        let granted = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }

        guard granted else {
            throw SpeechRecognitionError.microphoneDenied
        }
    }

    private func startAudioRecognition(
        continuation: AsyncThrowingStream<SpeechRecognitionUpdate, Error>.Continuation
    ) throws {
        stopRecognition()

        guard let speechRecognizer = SFSpeechRecognizer(locale: locale) else {
            throw SpeechRecognitionError.recognizerUnavailable
        }

        guard speechRecognizer.isAvailable else {
            throw SpeechRecognitionError.recognizerUnavailable
        }
        self.speechRecognizer = speechRecognizer

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            throw SpeechRecognitionError.audioSessionFailed(error.localizedDescription)
        }

        let audioEngine = AVAudioEngine()
        let recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        recognitionRequest.shouldReportPartialResults = true

        if #available(iOS 16.0, *) {
            recognitionRequest.addsPunctuation = true
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            if let result {
                continuation.yield(SpeechRecognitionUpdate(
                    transcript: result.bestTranscription.formattedString,
                    isFinal: result.isFinal
                ))

                if result.isFinal {
                    continuation.finish()
                    self?.stopRecognition()
                }
            }

            if let error {
                continuation.finish(throwing: error)
                self?.stopRecognition()
            }
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            recognitionTask?.cancel()
            recognitionTask = nil
            throw SpeechRecognitionError.audioSessionFailed(error.localizedDescription)
        }

        self.audioEngine = audioEngine
        self.recognitionRequest = recognitionRequest
    }
}
#endif
