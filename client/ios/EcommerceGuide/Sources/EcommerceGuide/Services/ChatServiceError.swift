import Foundation

public enum ChatServiceError: LocalizedError, Equatable {
    case invalidResponse
    case malformedEvent(String)
    case missingFixture(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "The server returned an invalid response."
        case .malformedEvent(let event):
            "Could not parse stream event: \(event)"
        case .missingFixture(let name):
            "Missing fixture: \(name)"
        }
    }
}
