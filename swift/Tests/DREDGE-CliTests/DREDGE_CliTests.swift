import XCTest
@testable import DREDGECli

final class DREDGE_CliTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(DREDGECli.version, "0.1.0")
    }
    
    func testVersionFormat() {
        // Verify version follows semantic versioning format (X.Y.Z)
        let versionComponents = DREDGECli.version.split(separator: ".")
        XCTAssertEqual(versionComponents.count, 3, "Version should have 3 components (major.minor.patch)")
        
        // Each component should be a valid number
        for component in versionComponents {
            XCTAssertNotNil(Int(component), "Version component '\(component)' should be a number")
        }
    }
    
    func testTagline() {
        XCTAssertFalse(DREDGECli.tagline.isEmpty, "Tagline should not be empty")
        XCTAssertEqual(DREDGECli.tagline, "Digital memory must be human-reachable.")
    }
    
    func testRunMethod() {
        // Test that run method exists and is callable
        // This test verifies the method can be invoked without crashing
        XCTAssertNoThrow(DREDGECli.run())
    }
}
