import Vapor

/// Typed errors returned from bridge handlers.
enum BridgeError: Error {
  case permissionDenied(String)
  case notFound(String)
  case invalidInput(String)
  case operationFailed(String)
}

extension BridgeError: AbortError {
  var status: HTTPResponseStatus {
    switch self {
    case .permissionDenied: return .forbidden
    case .notFound: return .notFound
    case .invalidInput: return .badRequest
    case .operationFailed: return .internalServerError
    }
  }

  var reason: String {
    switch self {
    case .permissionDenied(let msg): return "Permission denied: \(msg)"
    case .notFound(let msg): return "Not found: \(msg)"
    case .invalidInput(let msg): return "Invalid input: \(msg)"
    case .operationFailed(let msg): return "Operation failed: \(msg)"
    }
  }
}
