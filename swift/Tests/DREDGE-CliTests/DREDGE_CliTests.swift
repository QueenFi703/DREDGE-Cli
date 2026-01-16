import XCTest
@testable import DREDGE_Cli

final class DREDGE_CliTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(DREDGECli.version, "0.1.0")
    }
}
