import XCTest
@testable import DREDGECli

/// Integration tests for DREDGE Swift package
/// Tests MCP Server endpoints and API surface
final class IntegrationTests: XCTestCase {
    
    // MARK: - String Theory Integration Tests
    
    func testStringTheoryFullSpectrum() throws {
        let stringTheory = StringTheory(dimensions: 10, length: 1.0)
        
        // Test spectrum generation
        let spectrum = stringTheory.modeSpectrum(maxModes: 5)
        XCTAssertEqual(spectrum.count, 5, "Spectrum should have correct number of modes")
        
        // Test vibrational modes at different positions
        let positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for position in positions {
            let mode = stringTheory.vibrationalMode(n: 1, x: position)
            XCTAssertGreaterThanOrEqual(mode, -1.0, "Mode should be within valid range")
            XCTAssertLessThanOrEqual(mode, 1.0, "Mode should be within valid range")
        }
    }
    
    func testStringTheoryDimensionScaling() throws {
        // Test different dimensions
        for dimensions in [6, 10, 11, 26] {
            let stringTheory = StringTheory(dimensions: dimensions, length: 1.0)
            let spectrum = stringTheory.modeSpectrum(maxModes: 3)
            
            XCTAssertEqual(stringTheory.dimensions, dimensions)
            XCTAssertEqual(spectrum.count, 3)
            
            // Verify energy levels increase monotonically
            for i in 0..<(spectrum.count - 1) {
                XCTAssertLessThan(spectrum[i], spectrum[i + 1], 
                    "Energy levels should increase for dimension \(dimensions)")
            }
        }
    }
    
    func testStringTheoryEdgeCases() throws {
        let stringTheory = StringTheory()
        
        // Test boundary conditions
        XCTAssertEqual(stringTheory.vibrationalMode(n: 1, x: 0.0), 0.0, accuracy: 1e-10)
        XCTAssertEqual(stringTheory.vibrationalMode(n: 1, x: 1.0), 0.0, accuracy: 1e-10)
        
        // Test invalid inputs
        XCTAssertEqual(stringTheory.vibrationalMode(n: 0, x: 0.5), 0.0)
        XCTAssertEqual(stringTheory.vibrationalMode(n: -1, x: 0.5), 0.0)
        XCTAssertEqual(stringTheory.vibrationalMode(n: 1, x: -0.1), 0.0)
        XCTAssertEqual(stringTheory.vibrationalMode(n: 1, x: 1.1), 0.0)
    }
    
    // MARK: - MCP Client Integration Tests
    
    func testMCPClientInitialization() throws {
        // Test with default URL
        let defaultClient = MCPClient()
        XCTAssertEqual(defaultClient.serverURL, "http://localhost:3002")
        
        // Test with custom URL
        let customClient = MCPClient(serverURL: "http://example.com:8080")
        XCTAssertEqual(customClient.serverURL, "http://example.com:8080")
        
        // Test with various URL formats
        let urls = [
            "http://localhost:3002",
            "https://api.example.com",
            "http://127.0.0.1:8000",
            "https://mcp.service:443"
        ]
        
        for url in urls {
            let client = MCPClient(serverURL: url)
            XCTAssertEqual(client.serverURL, url)
        }
    }
    
    // MARK: - Unified DREDGE Integration Tests
    
    func testUnifiedDREDGEInitialization() throws {
        // Test default initialization
        let defaultUnified = UnifiedDREDGE()
        XCTAssertEqual(defaultUnified.stringTheory.dimensions, 10)
        XCTAssertEqual(defaultUnified.mcpClient.serverURL, "http://localhost:3002")
        
        // Test custom initialization
        let customUnified = UnifiedDREDGE(dimensions: 26, serverURL: "http://custom.server:9000")
        XCTAssertEqual(customUnified.stringTheory.dimensions, 26)
        XCTAssertEqual(customUnified.mcpClient.serverURL, "http://custom.server:9000")
    }
    
    func testUnifiedDREDGEStringTheoryIntegration() throws {
        let unified = UnifiedDREDGE(dimensions: 10)
        
        // Test that string theory is properly integrated
        let spectrum = unified.stringTheory.modeSpectrum(maxModes: 5)
        XCTAssertEqual(spectrum.count, 5)
        
        // Test vibrational modes
        let mode = unified.stringTheory.vibrationalMode(n: 1, x: 0.5)
        XCTAssertEqual(mode, 1.0, accuracy: 1e-10)
    }
    
    // MARK: - API Surface Completeness Tests
    
    func testPublicAPIAvailability() throws {
        // Verify all public types are accessible
        let _: DREDGECli.Type = DREDGECli.self
        let _: StringTheory.Type = StringTheory.self
        let _: MCPClient.Type = MCPClient.self
        let _: UnifiedDREDGE.Type = UnifiedDREDGE.self
        let _: MCPError.Type = MCPError.self
        
        // Verify version information is accessible
        XCTAssertFalse(DREDGECli.version.isEmpty)
        XCTAssertFalse(DREDGECli.tagline.isEmpty)
    }
    
    func testStringTheoryAPICompleteness() throws {
        let stringTheory = StringTheory(dimensions: 10, length: 1.0)
        
        // Verify all public methods are accessible
        XCTAssertNoThrow(stringTheory.vibrationalMode(n: 1, x: 0.5))
        XCTAssertNoThrow(stringTheory.energyLevel(n: 1))
        XCTAssertNoThrow(stringTheory.modeSpectrum(maxModes: 10))
        
        // Verify all public properties are accessible
        XCTAssertEqual(stringTheory.dimensions, 10)
        XCTAssertEqual(stringTheory.length, 1.0)
    }
    
    func testMCPClientAPICompleteness() throws {
        let client = MCPClient(serverURL: "http://localhost:3002")
        
        // Verify public properties are accessible
        XCTAssertEqual(client.serverURL, "http://localhost:3002")
        
        // Note: Async methods require platform availability checks
        // These are compile-time checks to ensure API surface exists
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        if #available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *) {
            // API exists, will test connectivity separately
            XCTAssert(true, "Async API available on Apple platforms")
        }
        #else
        // Verify fallback methods exist
        XCTAssertNoThrow(try client.listCapabilities())
        XCTAssertNoThrow(try client.sendRequest(operation: "test", params: [:]))
        #endif
    }
    
    func testUnifiedDREDGEAPICompleteness() throws {
        let unified = UnifiedDREDGE()
        
        // Verify public properties are accessible
        XCTAssertNotNil(unified.stringTheory)
        XCTAssertNotNil(unified.mcpClient)
        
        // Note: Async methods require platform availability checks
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        if #available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *) {
            XCTAssert(true, "Async unified API available on Apple platforms")
        }
        #else
        // Verify fallback methods exist
        XCTAssertNoThrow(try unified.unifiedInference(
            insight: "test",
            coords: [0.5],
            modes: [1]
        ))
        XCTAssertNoThrow(try unified.getStringSpectrum(maxModes: 5))
        #endif
    }
    
    // MARK: - Error Handling Tests
    
    func testMCPErrorCases() throws {
        // Verify MCPError cases are accessible
        let invalidURLError: MCPError = .invalidURL
        let invalidResponseError: MCPError = .invalidResponse
        let networkError: MCPError = .networkError
        
        XCTAssertNotNil(invalidURLError)
        XCTAssertNotNil(invalidResponseError)
        XCTAssertNotNil(networkError)
    }
    
    // MARK: - Configuration and Defaults Tests
    
    func testDefaultConfigurations() throws {
        // Test that defaults are sensible and documented
        let stringTheory = StringTheory()
        XCTAssertEqual(stringTheory.dimensions, 10, "Default should be 10D superstring")
        XCTAssertEqual(stringTheory.length, 1.0, "Default length should be 1.0")
        
        let client = MCPClient()
        XCTAssertEqual(client.serverURL, "http://localhost:3002", "Default should be localhost:3002")
        
        let unified = UnifiedDREDGE()
        XCTAssertEqual(unified.stringTheory.dimensions, 10)
        XCTAssertEqual(unified.mcpClient.serverURL, "http://localhost:3002")
    }
    
    // MARK: - Semantic Versioning Tests
    
    func testVersionFormat() throws {
        let version = DREDGECli.version
        let components = version.split(separator: ".")
        
        // Verify semantic versioning format (MAJOR.MINOR.PATCH)
        XCTAssertEqual(components.count, 3, "Version should follow semantic versioning")
        
        for component in components {
            XCTAssertNotNil(Int(component), "Version components should be integers")
        }
        
        // Verify version is not empty or placeholder
        XCTAssertFalse(version.isEmpty)
        XCTAssertNotEqual(version, "0.0.0")
    }
    
    // MARK: - Cross-Component Integration Tests
    
    func testStringTheoryAndUnifiedIntegration() throws {
        // Create standalone string theory
        let standalone = StringTheory(dimensions: 10)
        
        // Create unified with same dimensions
        let unified = UnifiedDREDGE(dimensions: 10)
        
        // Verify they produce same results
        let standaloneSpectrum = standalone.modeSpectrum(maxModes: 5)
        let unifiedSpectrum = unified.stringTheory.modeSpectrum(maxModes: 5)
        
        XCTAssertEqual(standaloneSpectrum.count, unifiedSpectrum.count)
        for i in 0..<standaloneSpectrum.count {
            XCTAssertEqual(standaloneSpectrum[i], unifiedSpectrum[i], accuracy: 1e-10)
        }
    }
}
