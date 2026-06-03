import XCTest
@testable import DREDGE

final class DREDGETests: XCTestCase {
    func testRuntimeIdentity() {
        XCTAssertEqual(DREDGERuntime.identity(), "DREDGE Swift runtime")
    }
}
