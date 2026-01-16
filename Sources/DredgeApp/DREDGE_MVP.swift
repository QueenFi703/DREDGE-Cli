// DREDGE – Distill, Recall, Emerge, Detect, Guide, Evolve
// MVP iOS Dredge Agent
// SwiftUI + Background Tasks + Voice + Lock Screen Widget

import SwiftUI
import BackgroundTasks
import DredgeCore

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

