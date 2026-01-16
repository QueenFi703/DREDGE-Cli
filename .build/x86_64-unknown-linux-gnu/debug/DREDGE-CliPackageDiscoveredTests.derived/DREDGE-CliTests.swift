import XCTest
@testable import DREDGE_CliTests

fileprivate extension DREDGECliTests {
    @available(*, deprecated, message: "Not actually deprecated. Marked as deprecated to allow inclusion of deprecated tests (which test deprecated functionality) without warnings")
    static nonisolated(unsafe) let __allTests__DREDGECliTests = [
        ("testExample", testExample)
    ]
}
@available(*, deprecated, message: "Not actually deprecated. Marked as deprecated to allow inclusion of deprecated tests (which test deprecated functionality) without warnings")
func __DREDGE_CliTests__allTests() -> [XCTestCaseEntry] {
    return [
        testCase(DREDGECliTests.__allTests__DREDGECliTests)
    ]
}