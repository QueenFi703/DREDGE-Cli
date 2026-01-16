import XCTest
@testable import DREDGE_Cli

final class DREDGE_CliTests: XCTestCase {
    func testExample() {
        // This is a placeholder test to ensure the test target builds
        XCTAssertTrue(true)
    }
    
    func testVersionExists() {
        // Verify the version constant is accessible and not empty
        XCTAssertFalse(DREDGECli.version.isEmpty)
    }
}
