import XCTest
@testable import DREDGECli

/// End-to-end tests for DREDGE-Cli
/// These tests require a running MCP server and test real network interactions
/// Run these tests manually when MCP server is available
final class EndToEndTests: XCTestCase {
    
    let testServerURL = "http://localhost:3002"
    var isServerAvailable = false
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Check if server is available before running tests
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        if #available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *) {
            let client = MCPClient(serverURL: testServerURL)
            do {
                _ = try await client.listCapabilities()
                isServerAvailable = true
            } catch {
                isServerAvailable = false
                print("⚠️  MCP Server not available at \(testServerURL). Skipping end-to-end tests.")
                print("   Start server with: python -m dredge mcp")
            }
        }
        #endif
    }
    
    // MARK: - MCP Server Connectivity Tests
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testMCPServerCapabilities() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let client = MCPClient(serverURL: testServerURL)
        let capabilities = try await client.listCapabilities()
        
        // Verify response structure
        XCTAssertNotNil(capabilities["name"])
        XCTAssertNotNil(capabilities["version"])
        XCTAssertNotNil(capabilities["capabilities"])
        
        // Log capabilities for visibility
        if let name = capabilities["name"] as? String {
            print("✓ Connected to: \(name)")
        }
        if let version = capabilities["version"] as? String {
            print("✓ Server version: \(version)")
        }
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testLoadQuasimodel() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let client = MCPClient(serverURL: testServerURL)
        let params: [String: Any] = ["model_type": "quasimoto_1d"]
        
        let response = try await client.sendRequest(operation: "load_model", params: params)
        
        // Verify response indicates success
        XCTAssertNotNil(response["success"])
        if let success = response["success"] as? Bool {
            XCTAssertTrue(success, "Model loading should succeed")
        }
        
        // Verify model_id is returned
        XCTAssertNotNil(response["model_id"])
        
        print("✓ Loaded model: \(response["model_type"] ?? "unknown")")
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testStringSpectrumOperation() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let client = MCPClient(serverURL: testServerURL)
        let params: [String: Any] = [
            "max_modes": 10,
            "dimensions": 10
        ]
        
        let response = try await client.sendRequest(operation: "string_spectrum", params: params)
        
        // Verify response
        XCTAssertNotNil(response["success"])
        if let success = response["success"] as? Bool {
            XCTAssertTrue(success, "String spectrum operation should succeed")
        }
        
        XCTAssertNotNil(response["energy_spectrum"])
        
        print("✓ String spectrum computed successfully")
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    // MARK: - Unified DREDGE End-to-End Tests
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testUnifiedInferenceEndToEnd() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let unified = UnifiedDREDGE(serverURL: testServerURL)
        
        let result = try await unified.unifiedInference(
            insight: "Digital memory must be human-reachable",
            coords: [0.5, 0.5, 0.5],
            modes: [1, 2, 3]
        )
        
        // Verify response structure
        XCTAssertNotNil(result["success"])
        if let success = result["success"] as? Bool {
            XCTAssertTrue(success, "Unified inference should succeed")
        }
        
        XCTAssertNotNil(result["unified_field"])
        XCTAssertNotNil(result["dredge_insight"])
        
        print("✓ Unified inference completed successfully")
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testGetStringSpectrumEndToEnd() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let unified = UnifiedDREDGE(serverURL: testServerURL)
        let result = try await unified.getStringSpectrum(maxModes: 5)
        
        // Verify response
        XCTAssertNotNil(result["success"])
        if let success = result["success"] as? Bool {
            XCTAssertTrue(success, "String spectrum retrieval should succeed")
        }
        
        XCTAssertNotNil(result["energy_spectrum"])
        
        print("✓ String spectrum retrieved successfully")
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    // MARK: - Error Handling Tests
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testInvalidOperationHandling() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let client = MCPClient(serverURL: testServerURL)
        
        do {
            _ = try await client.sendRequest(operation: "invalid_operation", params: [:])
            XCTFail("Should throw error for invalid operation")
        } catch {
            // Expected to throw error
            print("✓ Invalid operation properly rejected")
        }
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testConnectionToInvalidServer() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        let client = MCPClient(serverURL: "http://invalid.server:9999")
        
        do {
            _ = try await client.listCapabilities()
            XCTFail("Should throw error for invalid server")
        } catch {
            // Expected to throw error
            print("✓ Invalid server connection properly handled")
        }
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    // MARK: - Performance Tests
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testStringSpectrumPerformance() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let unified = UnifiedDREDGE(serverURL: testServerURL)
        
        measure {
            let expectation = XCTestExpectation(description: "String spectrum computation")
            
            Task {
                do {
                    _ = try await unified.getStringSpectrum(maxModes: 10)
                    expectation.fulfill()
                } catch {
                    XCTFail("Performance test failed: \(error)")
                }
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
    
    // MARK: - Regression Tests
    
    @available(macOS 12.0, iOS 15.0, watchOS 8.0, tvOS 15.0, *)
    func testMultipleSequentialRequests() async throws {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
        guard isServerAvailable else {
            throw XCTSkip("MCP Server not available")
        }
        
        let client = MCPClient(serverURL: testServerURL)
        
        // Test that multiple sequential requests work correctly
        for i in 1...3 {
            let result = try await client.sendRequest(
                operation: "string_spectrum",
                params: ["max_modes": i * 3, "dimensions": 10]
            )
            
            if let success = result["success"] as? Bool {
                XCTAssertTrue(success, "Request \(i) should succeed")
            }
            
            print("✓ Sequential request \(i) completed")
        }
        #else
        throw XCTSkip("Network tests only available on Apple platforms")
        #endif
    }
}
