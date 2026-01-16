// DREDGE – Distill, Recall, Emerge, Detect, Guide, Evolve
// Core Processing Engine

import Foundation
import NaturalLanguage

public struct DredgeEngine {
    // Cache NLTagger instance to avoid repeated initialization overhead
    private static let sentimentTagger: NLTagger = {
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        return tagger
    }()
    
    public static func process(thoughts: [String]) -> String {
        guard !thoughts.isEmpty else { return "Still waters." }

        // Efficient string joining - joined() is optimized internally
        let text = thoughts.joined(separator: ". ")
        
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
