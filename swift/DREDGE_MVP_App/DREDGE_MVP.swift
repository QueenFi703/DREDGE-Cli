// DREDGE – Distill, Recall, Emerge, Detect, Guide, Evolve
// MVP iOS Dredge Agent
// SwiftUI + Background Tasks + Voice + Shared Storage

#if canImport(SwiftUI)
import SwiftUI
#if os(iOS)
import BackgroundTasks
import AVFoundation
import Speech
#endif
import NaturalLanguage

@main
struct DredgeApp: App {
    init() {
        #if os(iOS)
        registerBackgroundTasks()
        scheduleNextProcessing()
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    #if os(iOS)
    private func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.dredge.agent.process",
            using: nil
        ) { task in
            guard let processingTask = task as? BGProcessingTask else {
                task.setTaskCompleted(success: false)
                return
            }
            handleProcessingTask(task: processingTask)
        }
    }

    private func handleProcessingTask(task: BGProcessingTask) {
        scheduleNextProcessing()

        let operation = DredgeOperation()
        task.expirationHandler = {
            operation.cancel()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.addOperation(operation)
    }

    private func scheduleNextProcessing() {
        let request = BGProcessingTaskRequest(identifier: "com.dredge.agent.process")
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            // Scheduling is best-effort; iOS controls the actual execution time.
        }
    }
    #endif
}

// MARK: - Core UI

struct ContentView: View {
    @State private var thoughts: [String] = SharedStore.loadThoughts()
    @State private var surfacedInsight: String = SharedStore.loadSurfaced()
    @State private var isRecording = false
    @State private var voiceError: String?
    @State private var voiceDredger = VoiceDredger()

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("DREDGE")
                    .font(.largeTitle)
                    .fontWeight(.semibold)

                Button(isRecording ? "Stop Listening" : "Voice Dredge") {
                    toggleRecording()
                }

                Button("Process") {
                    processThoughts()
                }

                Text(surfacedInsight)
                    .italic()
                    .padding()

                if let voiceError {
                    Text(voiceError)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                List(thoughts, id: \.self) { thought in
                    Text(thought)
                }
            }
            .padding()
            .navigationTitle("Collected")
        }
    }

    private func processThoughts() {
        surfacedInsight = DredgeEngine.process(thoughts: thoughts)
        SharedStore.saveSurfaced(surfacedInsight)
    }

    private func toggleRecording() {
        voiceError = nil

        if isRecording {
            voiceDredger.stop()
            if let result = voiceDredger.latestTranscription {
                let trimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    thoughts.append(trimmed)
                    SharedStore.appendThought(trimmed)
                }
            }
            isRecording = false
            return
        }

        voiceDredger.start { result in
            switch result {
            case .success:
                isRecording = true
            case .failure(let error):
                voiceError = error.localizedDescription
                isRecording = false
            }
        }
    }
}

// MARK: - Dredge Engine

struct DredgeEngine {
    private static let sentimentTagger: NLTagger = {
        NLTagger(tagSchemes: [.sentimentScore])
    }()

    static func process(thoughts: [String]) -> String {
        let cleaned = thoughts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !cleaned.isEmpty else { return "Still waters." }

        let text = cleaned.joined(separator: ". ")
        sentimentTagger.string = text

        let sentiment = sentimentTagger.tag(
            at: text.startIndex,
            unit: .paragraph,
            scheme: .sentimentScore
        ).0

        let score = Double(sentiment?.rawValue ?? "0") ?? 0

        switch score {
        case let s where s > 0.3:
            return "A gentle clarity is forming."
        case let s where s < -0.3:
            return "Something beneath asks for rest."
        default:
            return "Balance holds."
        }
    }
}

#if os(iOS)
// MARK: - Voice Dredger

final class VoiceDredger {
    enum VoiceError: LocalizedError {
        case speechNotAuthorized
        case recognizerUnavailable
        case microphoneUnavailable
        case audioSessionFailed
        case recordingFailed

        var errorDescription: String? {
            switch self {
            case .speechNotAuthorized:
                return "Speech recognition permission is required for Voice Dredge."
            case .recognizerUnavailable:
                return "Speech recognition is currently unavailable."
            case .microphoneUnavailable:
                return "Microphone input is unavailable on this device."
            case .audioSessionFailed:
                return "DREDGE could not configure the microphone."
            case .recordingFailed:
                return "DREDGE could not start voice capture."
            }
        }
    }

    private let audioEngine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer(locale: Locale.current)
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private let bufferSize: AVAudioFrameCount

    var latestTranscription: String?

    init(bufferSize: AVAudioFrameCount = 1024) {
        self.bufferSize = bufferSize
    }

    func start(completion: @escaping (Result<Void, VoiceError>) -> Void) {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            SFSpeechRecognizer.requestAuthorization { status in
                DispatchQueue.main.async {
                    guard status == .authorized else {
                        completion(.failure(.speechNotAuthorized))
                        return
                    }
                    self.start(completion: completion)
                }
            }
            return
        }

        guard let recognizer, recognizer.isAvailable else {
            completion(.failure(.recognizerUnavailable))
            return
        }

        stop()
        latestTranscription = nil

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            completion(.failure(.audioSessionFailed))
            return
        }

        guard audioSession.recordPermission == .granted else {
            audioSession.requestRecordPermission { granted in
                DispatchQueue.main.async {
                    guard granted else {
                        completion(.failure(.microphoneUnavailable))
                        return
                    }
                    self.start(completion: completion)
                }
            }
            return
        }

        guard audioEngine.inputNode.inputFormat(forBus: 0).channelCount > 0 else {
            completion(.failure(.microphoneUnavailable))
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        self.request = request

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: format) { buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()

        do {
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            self.request = nil
            try? audioSession.setActive(false, options: .notifyOthersOnDeactivation)
            completion(.failure(.recordingFailed))
            return
        }

        task = recognizer.recognitionTask(with: request) { [weak self] result, _ in
            guard let result else { return }
            let transcription = result.bestTranscription.formattedString
            DispatchQueue.main.async {
                self?.latestTranscription = transcription
            }
        }

        completion(.success(()))
    }

    func stop() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        request = nil
        task?.cancel()
        task = nil
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }
}

// MARK: - Background Operation

final class DredgeOperation: Operation {
    override func main() {
        guard !isCancelled else { return }

        let thoughts = SharedStore.loadThoughts()
        guard !thoughts.isEmpty else { return }

        let insight = DredgeEngine.process(thoughts: thoughts)
        guard !isCancelled else { return }

        SharedStore.saveSurfaced(insight)
    }
}
#endif
#endif
