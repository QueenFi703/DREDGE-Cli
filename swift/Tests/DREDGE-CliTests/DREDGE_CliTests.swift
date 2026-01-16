import XCTest
@testable import DREDGECli

final class DREDGE_CliTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(DREDGECli.version, "0.1.0")
    }
}
