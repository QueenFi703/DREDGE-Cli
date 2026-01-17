import Foundation

// MARK: - Structured Logging

/// Log levels for structured logging
public enum LogLevel: String {
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARNING"
    case error = "ERROR"
}

/// Structured logger for DREDGE-Cli
public struct Logger {
    private let component: String
    private let enabled: Bool
    
    public init(component: String, enabled: Bool = true) {
        self.component = component
        self.enabled = enabled
    }
    
    public func log(_ level: LogLevel, _ message: String, context: [String: Any] = [:]) {
        guard enabled else { return }
        
        let timestamp = ISO8601DateFormatter().string(from: Date())
        var logMessage = "[\(timestamp)] [\(level.rawValue)] [\(component)] \(message)"
        
        if !context.isEmpty {
            let contextStr = context.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            logMessage += " | \(contextStr)"
        }
        
        print(logMessage)
    }
    
    public func debug(_ message: String, context: [String: Any] = [:]) {
        log(.debug, message, context: context)
    }
    
    public func info(_ message: String, context: [String: Any] = [:]) {
        log(.info, message, context: context)
    }
    
    public func warning(_ message: String, context: [String: Any] = [:]) {
        log(.warning, message, context: context)
    }
    
    public func error(_ message: String, context: [String: Any] = [:]) {
        log(.error, message, context: context)
    }
}

// MARK: - DREDGE CLI

public struct DREDGECli {
    public static let version = "0.2.0"
    public static let tagline = "Digital memory must be human-reachable."
    
    private static let logger = Logger(component: "DREDGECli")
    
    public static func run() {
        logger.info("Starting DREDGE-Cli", context: ["version": version])
        
        print("DREDGE-Cli v\(version)")
        print(tagline)
        print("")
        print("Features:")
        print("  • DREDGE x Dolly Integration")
        print("  • Quasimoto Wave Functions (1D, 4D, 6D, Ensemble)")
        print("  • String Theory Models (10D Superstring)")
        print("  • Unified MCP Server")
        print("")
        print("Run 'dredge-cli help' for more information")
        
        logger.debug("CLI information displayed")
    }
}

// MARK: - String Theory Support

public struct StringTheory {
    public let dimensions: Int
    public let length: Double
    private let logger = Logger(component: "StringTheory")
    
    public init(dimensions: Int = 10, length: Double = 1.0) {
        self.dimensions = dimensions
        self.length = length
        logger.info("Initialized StringTheory", context: [
            "dimensions": dimensions,
            "length": length
        ])
    }
    
    /// Calculate the nth vibrational mode at position x
    /// Returns 0.0 for invalid inputs (consistent with error handling)
    public func vibrationalMode(n: Int, x: Double) -> Double {
        guard n >= 1 else {
            logger.warning("Invalid mode number", context: ["n": n, "expected": ">=1"])
            return 0.0
        }
        guard x >= 0.0 && x <= 1.0 else {
            logger.warning("Invalid position", context: ["x": x, "expected": "0.0...1.0"])
            return 0.0
        }
        
        let result = sin(Double(n) * Double.pi * x)
        logger.debug("Calculated vibrational mode", context: [
            "n": n,
            "x": x,
            "result": result
        ])
        return result
    }
    
    /// Calculate energy level for the nth mode
    public func energyLevel(n: Int) -> Double {
        let energy = Double(n) / (2.0 * length)
        logger.debug("Calculated energy level", context: [
            "n": n,
            "energy": energy
        ])
        return energy
    }
    
    /// Generate energy spectrum
    public func modeSpectrum(maxModes: Int = 10) -> [Double] {
        logger.info("Generating mode spectrum", context: ["maxModes": maxModes])
        let spectrum = (1...maxModes).map { energyLevel(n: $0) }
        logger.debug("Generated spectrum", context: [
            "count": spectrum.count,
            "first": spectrum.first ?? 0.0,
            "last": spectrum.last ?? 0.0
        ])
        return spectrum
    }
}

// MARK: - MCP Client Support

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct MCPClient {
    public let serverURL: String
    private static let mcpEndpoint = "/mcp"
    private let logger = Logger(component: "MCPClient")
    
    public init(serverURL: String = "http://localhost:3002") {
        self.serverURL = serverURL
        logger.info("Initialized MCPClient", context: ["serverURL": serverURL])
    }
    
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
    /// List available MCP capabilities (Apple platforms)
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    public func listCapabilities() async throws -> [String: Any] {
        logger.info("Listing MCP capabilities")
        
        guard let url = URL(string: serverURL) else {
            logger.error("Invalid URL", context: ["serverURL": serverURL])
            throw MCPError.invalidURL
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            if let httpResponse = response as? HTTPURLResponse {
                logger.debug("HTTP response", context: ["statusCode": httpResponse.statusCode])
            }
            
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                logger.error("Invalid JSON response")
                throw MCPError.invalidResponse
            }
            
            logger.info("Retrieved capabilities", context: [
                "keys": Array(json.keys).joined(separator: ", ")
            ])
            return json
        } catch let error as MCPError {
            throw error
        } catch {
            logger.error("Network error", context: ["error": error.localizedDescription])
            throw MCPError.networkError
        }
    }
    
    /// Send MCP request (Apple platforms)
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    public func sendRequest(operation: String, params: [String: Any]) async throws -> [String: Any] {
        logger.info("Sending MCP request", context: [
            "operation": operation,
            "params": params.description
        ])
        
        guard let url = URL(string: "\(serverURL)\(MCPClient.mcpEndpoint)") else {
            logger.error("Invalid URL", context: ["serverURL": serverURL])
            throw MCPError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "operation": operation,
            "params": params
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            if let httpResponse = response as? HTTPURLResponse {
                logger.debug("HTTP response", context: [
                    "statusCode": httpResponse.statusCode,
                    "operation": operation
                ])
            }
            
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                logger.error("Invalid JSON response", context: ["operation": operation])
                throw MCPError.invalidResponse
            }
            
            if let success = json["success"] as? Bool {
                let level: LogLevel = success ? .info : .warning
                logger.log(level, "MCP operation completed", context: [
                    "operation": operation,
                    "success": success
                ])
            }
            
            return json
        } catch let error as MCPError {
            throw error
        } catch {
            logger.error("Network error", context: [
                "operation": operation,
                "error": error.localizedDescription
            ])
            throw MCPError.networkError
        }
    }
    #else
    /// Placeholder for non-Apple platforms (Linux, etc.)
    public func listCapabilities() throws -> [String: Any] {
        logger.warning("MCP Client networking not available on this platform")
        print("Note: MCP Client networking requires macOS 12.0+ or iOS 15.0+")
        return ["note": "MCP Client networking not available on this platform"]
    }
    
    /// Placeholder for non-Apple platforms (Linux, etc.)
    public func sendRequest(operation: String, params: [String: Any]) throws -> [String: Any] {
        logger.warning("MCP Client networking not available on this platform", context: [
            "operation": operation
        ])
        print("Note: MCP Client networking requires macOS 12.0+ or iOS 15.0+")
        return ["note": "MCP Client networking not available on this platform"]
    }
    #endif
}

public enum MCPError: Error {
    case invalidURL
    case invalidResponse
    case networkError
}

// MARK: - Unified Integration

public struct UnifiedDREDGE {
    public let stringTheory: StringTheory
    public let mcpClient: MCPClient
    private let logger = Logger(component: "UnifiedDREDGE")
    
    public init(dimensions: Int = 10, serverURL: String = "http://localhost:3002") {
        self.stringTheory = StringTheory(dimensions: dimensions)
        self.mcpClient = MCPClient(serverURL: serverURL)
        logger.info("Initialized UnifiedDREDGE", context: [
            "dimensions": dimensions,
            "serverURL": serverURL
        ])
    }
    
    #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
    /// Compute unified field combining string theory and Quasimoto (Apple platforms)
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    public func unifiedInference(
        insight: String,
        coords: [Double],
        modes: [Int]
    ) async throws -> [String: Any] {
        logger.info("Starting unified inference", context: [
            "insight": insight.prefix(50).description,
            "coords_count": coords.count,
            "modes_count": modes.count
        ])
        
        let params: [String: Any] = [
            "dredge_insight": insight,
            "quasimoto_coords": coords,
            "string_modes": modes
        ]
        
        let result = try await mcpClient.sendRequest(operation: "unified_inference", params: params)
        
        if let success = result["success"] as? Bool, success {
            logger.info("Unified inference completed successfully")
        } else {
            logger.warning("Unified inference failed or returned error")
        }
        
        return result
    }
    
    /// Get string theory spectrum (Apple platforms)
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    public func getStringSpectrum(maxModes: Int = 10) async throws -> [String: Any] {
        logger.info("Getting string spectrum", context: ["maxModes": maxModes])
        
        let params: [String: Any] = [
            "max_modes": maxModes,
            "dimensions": stringTheory.dimensions
        ]
        
        let result = try await mcpClient.sendRequest(operation: "string_spectrum", params: params)
        
        logger.debug("String spectrum retrieved", context: [
            "maxModes": maxModes,
            "dimensions": stringTheory.dimensions
        ])
        
        return result
    }
    #else
    /// Placeholder for non-Apple platforms
    public func unifiedInference(
        insight: String,
        coords: [Double],
        modes: [Int]
    ) throws -> [String: Any] {
        logger.warning("Unified inference networking not available on this platform")
        print("Note: Unified inference networking requires macOS 12.0+ or iOS 15.0+")
        return ["note": "Networking not available on this platform"]
    }
    
    /// Placeholder for non-Apple platforms
    public func getStringSpectrum(maxModes: Int = 10) throws -> [String: Any] {
        logger.warning("String spectrum networking not available on this platform")
        print("Note: String spectrum networking requires macOS 12.0+ or iOS 15.0+")
        return ["note": "Networking not available on this platform"]
    }
    #endif
}

DREDGECli.run()
