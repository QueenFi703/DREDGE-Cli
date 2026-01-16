import Foundation
public enum SharedStore {
    public static let appGroupID = "group.com.dredge.agent"
    public static let surfacedKey = "surfacedInsight"
    public static let fileName = "surfaced_insight.txt"
    public static var defaults: UserDefaults? { UserDefaults(suiteName: appGroupID) }
    public static var containerURL: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupID)
    }
    public static var iCloudURL: URL? {
        guard FileManager.default.ubiquityIdentityToken != nil else { return nil }
        return FileManager.default.url(forUbiquityContainerIdentifier: nil)?.appendingPathComponent("Documents")
    }
    public static func surfacedFileURL() -> URL? {
        containerURL?.appendingPathComponent(fileName)
    }
    public static func saveSurfaced(_ text: String) {
        defaults?.set(text, forKey: surfacedKey)
        saveToFile(text)
        DispatchQueue.global(qos: .background).async { syncToICloud() }
    }
    public static func saveToFile(_ text: String) {
        guard let url = surfacedFileURL() else { return }
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
    public static func loadSurfaced() -> String {
        if let url = surfacedFileURL(),
           let text = try? String(contentsOf: url) { return text }
        return defaults?.string(forKey: surfacedKey) ?? "Nothing surfaced yet."
    }
    public static func syncToICloud() {
        guard let localURL = surfacedFileURL(),
              let cloudDir = iCloudURL else { return }
        let cloudURL = cloudDir.appendingPathComponent(fileName)
        try? FileManager.default.createDirectory(at: cloudDir, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: cloudURL)
        try? FileManager.default.copyItem(at: localURL, to: cloudURL)
    }
}
