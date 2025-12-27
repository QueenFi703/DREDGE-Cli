// DREDGE – Distill, Recall, Emerge, Detect, Guide, Evolve
// MVP iOS Dredge Agent
// SwiftUI + Background Tasks + Voice + Lock Screen Widget

import SwiftUI
import BackgroundTasks
import NaturalLanguage
import Speech
import AVFoundation

@main
struct DredgeApp: App {
    init() {
        registerBackgroundTasks()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    private func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: "com.dredge.agent.process",
            using: nil
        ) { task in
            handleProcessingTask(task: task as! BGProcessingTask)
        }
    }

    private func handleProcessingTask(task: BGProcessingTask) {
        scheduleNextProcessing()

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        let operation = DredgeOperation()

        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }

    private func scheduleNextProcessing() {
        let request = BGProcessingTaskRequest(identifier: "com.dredge.agent.process")
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false

        try? BGTaskScheduler.shared.submit(request)
    }
}

// MARK: - Core UI

struct ContentView: View {
    @State private var thoughts: [String] = []
    @State private var surfacedInsight: String = "Nothing surfaced yet."
    @State private var isRecording = false

    private let voiceDredger = VoiceDredger()

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
                    surfacedInsight = DredgeEngine.process(thoughts: thoughts)
                }

                Text(surfacedInsight)
                    .italic()
                    .padding()

                List(thoughts, id: \.self) { thought in
                    Text(thought)
                }
            }
            .padding()
            .navigationTitle("Collected")
        }
    }

    private func toggleRecording() {
        if isRecording {
            voiceDredger.stop()
            if let result = voiceDredger.latestTranscription {
                thoughts.append(result)
            }
        } else {
            voiceDredger.start()
        }
        isRecording.toggle()
    }
}

// MARK: - Dredge Engine

struct DredgeEngine {
    static func process(thoughts: [String]) -> String {
        guard !thoughts.isEmpty else { return "Still waters." }

        let text = thoughts.joined(separator: ". ")
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text

        let sentiment = tagger.tag(
            at: text.startIndex,
            unit: .paragraph,
            scheme: .sentimentScore
        ).0

        let score = Double(sentiment?.rawValue ?? "0") ?? 0

        switch score {
        case let sentimentScore where sentimentScore > 0.3:
            return "A gentle clarity is forming."
        case let sentimentScore where sentimentScore < -0.3:
            return "Something beneath asks for rest."
        default:
            return "Balance holds."
        }
    }
}

// MARK: - Voice Dredger

final class VoiceDredger {
    private let audioEngine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    var latestTranscription: String?

    init() {
        SFSpeechRecognizer.requestAuthorization { _ in }
    }

    func start() {
        latestTranscription = nil
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) {
            buffer, _ in recognitionRequest.append(buffer)
        }

        audioEngine.prepare()
        try? audioEngine.start()

        recognitionTask = recognizer?.recognitionTask(with: recognitionRequest) { result, _ in
            if let result = result {
                self.latestTranscription = result.bestTranscription.formattedString
            }
        }
    }

    func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
    }
}

// MARK: - Background Operation

class DredgeOperation: Operation {
    override func main() {
        if isCancelled { return }
        sleep(2)
    }
}
