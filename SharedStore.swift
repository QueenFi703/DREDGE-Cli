import Foundation

public enum SharedStore {
    private static let insightStorageKey = "surfacedInsight"
    private static let sharedDefaults = UserDefaults(suiteName: "group.com.dredge.agent")
    public static func saveSurfaced(_ text: String) { sharedDefaults?.set(text, forKey: insightStorageKey) }
    public static func loadSurfaced() -> String { sharedDefaults?.string(forKey: insightStorageKey) ?? "Something surfaced…" }
}
