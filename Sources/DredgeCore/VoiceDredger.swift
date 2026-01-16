// DREDGE – Distill, Recall, Emerge, Detect, Guide, Evolve
// Voice Recognition Module

import Foundation
import Speech
import AVFoundation

public final class VoiceDredger {
    private let audioEngine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    
    // Configurable buffer size for performance tuning (default: 1024)
    // Larger buffers reduce CPU overhead but increase latency
    private let bufferSize: AVAudioFrameCount

    public var latestTranscription: String?

    public init(bufferSize: AVAudioFrameCount = 1024) {
        self.bufferSize = bufferSize
        SFSpeechRecognizer.requestAuthorization { _ in }
    }

    public func start() {
        latestTranscription = nil
        request = SFSpeechAudioBufferRecognitionRequest()
        guard let request = request else { return }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: format) {
            buffer, _ in request.append(buffer)
        }

        audioEngine.prepare()
        try? audioEngine.start()

        task = recognizer?.recognitionTask(with: request) { result, _ in
            if let result = result {
                self.latestTranscription = result.bestTranscription.formattedString
            }
        }
    }

    public func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
    }
}
