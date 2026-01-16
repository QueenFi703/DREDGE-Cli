import Foundation
public struct DredgeEngine {
    public static func process(_ thoughts: [String]) -> String {
        thoughts.isEmpty ? "Still waters." : "Balance holds."
    }
}
