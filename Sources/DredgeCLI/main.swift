import Foundation
import DredgeCore

@main
struct DREDGECli {
    static let version = "0.1.0"
    
    static func main() {
        print("DREDGE-Cli v\(version)")
        print("Digital memory must be human-reachable.")
        
        // Example: Process some thoughts using DredgeCore
        let thoughts = ["Digital memory", "Human-reachable systems"]
        let insight = DredgeEngine.process(thoughts: thoughts)
        print("Insight: \(insight)")
    }
}
