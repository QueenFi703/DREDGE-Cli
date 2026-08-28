import Foundation

public enum SharedStore {
    private static let suiteName = "group.com.dredge.agent"
    private static let surfacedInsightKey = "surfacedInsight"
    private static let thoughtsKey = "thoughts"

    private static let defaults = UserDefaults(suiteName: suiteName)

    public static func saveSurfaced(_ text: String) {
        defaults?.set(text, forKey: surfacedInsightKey)
    }

    public static func loadSurfaced() -> String {
        defaults?.string(forKey: surfacedInsightKey) ?? "Something surfaced…"
    }

    public static func saveThoughts(_ thoughts: [String]) {
        defaults?.set(thoughts, forKey: thoughtsKey)
    }

    public static func loadThoughts() -> [String] {
        defaults?.stringArray(forKey: thoughtsKey) ?? []
    }

    public static func appendThought(_ thought: String) {
        let trimmed = thought.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        saveThoughts(loadThoughts() + [trimmed])
    }
}
