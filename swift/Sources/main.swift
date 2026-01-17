import Foundation

public struct DREDGECli {
    public static let version = "0.1.0"
    public static let tagline = "Digital memory must be human-reachable."
    
    public static func run() {
        print("DREDGE-Cli v\(version)")
        print(tagline)
    }
}

DREDGECli.run()