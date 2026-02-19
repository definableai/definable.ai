import Vapor

/// Bearer token authentication middleware.
/// Every request must include: ``Authorization: Bearer <token>``
struct BearerAuthMiddleware: AsyncMiddleware {
  let expectedToken: String

  func respond(to request: Request, chainingTo next: AsyncResponder) async throws -> Response {
    guard let authHeader = request.headers.bearerAuthorization else {
      throw Abort(.unauthorized, reason: "Missing Authorization: Bearer header")
    }
    guard authHeader.token == expectedToken else {
      throw Abort(.forbidden, reason: "Invalid bearer token")
    }
    return try await next.respond(to: request)
  }
}
